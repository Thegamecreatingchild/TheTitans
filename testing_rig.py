"""
ball_follow.py — minimal camera-driven ball follower for the MHS RCJ Soccer bot.
 
What it does
------------
1. Grabs frames from the Pi camera, thresholds for orange in HSV, and finds the
   largest blob that is NOT inside the centre-ignore circle (the robot's own
   body / dribbler area). That blob is assumed to be the ball.
2. Converts the blob's pixel offset from image centre into a bearing.
3. Drives the holonomic base: rotate in place until the ball is within
   FORWARD_ALIGN_TOLERANCE of dead ahead, then drive straight at it.
   The bot always turns to *face* the ball — it never reverses toward it.
4. Serves an MJPEG stream plus live HSV / contour-area / centre-ignore sliders
   over HTTP so thresholds can be tuned at the venue without restarting.
 
Threads
-------
- main thread      : vision_loop()   — camera capture + detection (main thread so
                                       cv2.imshow works when SHOW_WINDOWS=True)
- motor_thread     : motor_loop()    — reads shared ball state, commands motors
- stream_thread    : web_stream_loop() — Flask MJPEG + slider endpoints
 
Shared state is protected by three locks: state_lock (ball position),
params_lock (detection thresholds), stream_lock (latest JPEG).
 
Hardware
--------
- 4× Steel Bar PowerfulBLDC drivers over I2C, addresses 25–28 = FL, FR, RL, RR
- Raspberry Pi camera via picamera2
"""
 
import math
import time
import threading
import signal
import sys
 
import cv2
import numpy as np
import board
import busio
from picamera2 import Picamera2
from steelbar_powerful_bldc_driver import PowerfulBLDCDriver
from flask import Flask, Response, request, jsonify
 
# ----------------------------------------------------------------
# Config
# ----------------------------------------------------------------
 
# I2C addresses of the four drive motor drivers, in the order [FL, FR, RL, RR].
# Index into this list == index into `motors` and SAVED_CAL.
MOTOR_ADDRESSES = [25, 26, 27, 28]
 
# Per-motor FOC calibration captured from a previous calibration run.
# Order matches MOTOR_ADDRESSES. Re-run calibration if a motor or driver is swapped.
SAVED_CAL = [
    {'elecangleoffset': 1327731200, 'sincoscentre': 1241},  # FL
    {'elecangleoffset': 1435147520, 'sincoscentre': 1243},  # FR
    {'elecangleoffset': 1256835584, 'sincoscentre': 1258},  # RL
    {'elecangleoffset': 1150337792, 'sincoscentre': 1247},  # RR
]
 
# Base speed command sent to the drivers (raw driver units, not RPM).
# Actual per-motor command = pattern_sign * SPEED * MOTOR_SPEED_SCALE[motor].
SPEED = 300000000
 
# Per-motor multiplier applied to SPEED. Use this to trim motors that spin
# faster/slower than the others so the bot tracks straight.
MOTOR_SPEED_SCALE = {
    'FL': 1.0,
    'FR': 0.50,
    'RL': 0.50,
    'RR': 0.50,
}
 
# Motor name → index into `motors` / MOTOR_ADDRESSES.
MOTOR_INDEX = {
    'FL': 0,
    'FR': 1,
    'RL': 2,
    'RR': 3,
}
 
# Direction sign for each motor per action. Signs encode both the mechanical
# mounting of each motor (left side is mirrored) and the wheel geometry.
# 'back' is kept for manual testing but is NOT used by the follow logic —
# the bot rotates to face the ball instead of reversing.
MOTOR_ACTIONS = {
    'forward': {'FL': -1, 'FR': 1,  'RL': -1, 'RR': 1},
    'back':    {'FL': 1,  'FR': -1, 'RL': 1,  'RR': -1},
    'cw':      {'FL': -1, 'FR': -1, 'RL': -1, 'RR': -1},
    'ccw':     {'FL': 1,  'FR': 1,  'RL': 1,  'RR': 1},
}
 
# Degrees added to the image-derived bearing so that 0° == robot forward.
# Accounts for the camera being mounted rotated relative to the chassis.
CAMERA_ROTATION_OFFSET = -90.0
 
# ================================================================
# TUNE-ME BLOCK - everything you'll adjust during testing lives here
# ================================================================
 
# Pixel radius around image centre inside which the ball is considered
# "reached" — motors stop so the bot doesn't hunt/oscillate on top of the ball.
DEAD_ZONE_RADIUS = 100
 
# Bearing error (degrees) within which the bot drives forward instead of
# rotating. Larger = drives sooner but tracks less precisely.
FORWARD_ALIGN_TOLERANCE = 15.0
 
# Seconds without a detection before the bot stops. Bridges single-frame
# dropouts without letting the bot keep driving on stale data.
BALL_LOST_TIMEOUT = 0.3
 
# Motor control loop rate.
MOTOR_LOOP_HZ = 50
 
# Open local cv2 windows (needs a desktop session). Off for headless runs.
SHOW_WINDOWS = False
 
# Serve the MJPEG stream + tuning sliders at http://<pi-ip>:STREAM_PORT
ENABLE_WEB_STREAM = True
STREAM_PORT = 5000
STREAM_JPEG_QUALITY = 80
# ================================================================
 
# ----------------------------------------------------------------
# Tunable detection parameters, adjustable live via sliders on the
# web page. vision_loop() reads a snapshot of this each frame -
# the sliders just write into it. Defaults match the last hardcoded
# values you were using.
# ----------------------------------------------------------------
params_lock = threading.Lock()
detect_params = {
    # HSV threshold bounds for the orange ball (OpenCV ranges: H 0-179, S/V 0-255).
    'h_low': 0,
    's_low': 132,
    'v_low': 166,
    'h_high': 10,
    's_high': 255,
    'v_high': 255,
    'min_contour_area': 30,       # px^2: smallest contour accepted as the ball
    'centre_ignore_radius': 60,   # px: contours whose centroid is closer than
                                  # this to image centre are ignored (own body,
                                  # dribbler, reflections). Must be < DEAD_ZONE_RADIUS
                                  # or the bot can never "arrive" at the ball.
}
 
# --- Shared ball state (vision writes, motor loop reads) ---
state_lock = threading.Lock()
ball_offset = None   # (dx, dy) pixels from image centre, or None if not seen
last_seen = 0.0      # time.time() of the last successful detection
isRunning = True     # global run flag; set False to stop all loops
 
motors = []          # list[PowerfulBLDCDriver], populated by setup_motors()
 
# --- Latest encoded frame for the MJPEG stream ---
stream_lock = threading.Lock()
latest_jpeg = None
 
 
# ----------------------------------------------------------------
# Motors
# ----------------------------------------------------------------
def setup_motors():
    """Initialise all four BLDC drivers over I2C and leave them stopped.
 
    Sets current limits, PID constants, restores the saved FOC calibration,
    selects operating mode 3 / sensor 1 and command mode 12 (speed control).
    """
    global motors
    i2c = busio.I2C(board.SCL, board.SDA)
 
    for index, address in enumerate(MOTOR_ADDRESSES):
        motor = PowerfulBLDCDriver(i2c, address)
 
        motor.set_current_limit_foc(65536)
        motor.set_id_pid_constants(1500, 200)
        motor.set_iq_pid_constants(1500, 200)
        motor.set_speed_pid_constants(4e-2, 4e-4, 3e-2)
 
        # Restore per-motor calibration so FOC commutation is correct.
        motor.set_ELECANGLEOFFSET(SAVED_CAL[index]["elecangleoffset"])
        motor.set_SINCOSCENTRE(SAVED_CAL[index]["sincoscentre"])
 
        motor.configure_operating_mode_and_sensor(3, 1)
        motor.configure_command_mode(12)
        motor.set_speed(0)
 
        motors.append(motor)
        print(f"Motor {index} ready")
 
    print("All motors ready")
 
 
def apply_action(action_name):
    """Command all four motors according to a MOTOR_ACTIONS pattern.
 
    Args:
        action_name: key into MOTOR_ACTIONS ('forward', 'cw', 'ccw', ...).
    """
    pattern = MOTOR_ACTIONS[action_name]
    for motor_name, index in MOTOR_INDEX.items():
        speed = int(pattern[motor_name] * SPEED * MOTOR_SPEED_SCALE[motor_name])
        motors[index].set_speed(speed)
 
 
def stop():
    """Set every motor's speed to zero."""
    for motor in motors:
        motor.set_speed(0)
 
 
def signed_heading_error(bearing):
    """Wrap a bearing in degrees to the range (-180, 180].
 
    Positive = ball is clockwise of forward, negative = counter-clockwise.
    """
    return ((bearing + 180) % 360) - 180
 
 
def choose_action(bearing):
    """Pick a drive action from the ball's bearing relative to robot forward.
 
    If the ball is within FORWARD_ALIGN_TOLERANCE of straight ahead, drive
    forward. Otherwise rotate in place toward the ball, choosing the shorter
    direction. The bot never reverses — a ball behind it is handled by turning
    around to face it.
 
    Args:
        bearing: degrees, 0 == robot forward, increasing clockwise.
    Returns:
        One of 'forward', 'cw', 'ccw'.
    """
    error = signed_heading_error(bearing)
 
    if abs(error) <= FORWARD_ALIGN_TOLERANCE:
        return 'forward'
 
    return 'cw' if error > 0 else 'ccw'
 
 
# ----------------------------------------------------------------
# Motor control loop (worker thread)
# ----------------------------------------------------------------
def motor_loop():
    """Poll the shared ball state at MOTOR_LOOP_HZ and drive the motors.
 
    Behaviour:
      - ball not seen for > BALL_LOST_TIMEOUT  → stop
      - ball inside DEAD_ZONE_RADIUS           → stop (arrived)
      - otherwise                              → rotate to face / drive forward
 
    `moving` tracks whether a stop() has already been issued so we don't
    hammer the I2C bus with redundant zero-speed commands.
    """
    period = 1.0 / MOTOR_LOOP_HZ
    moving = False
 
    while isRunning:
        with state_lock:
            offset = ball_offset
            seen = last_seen
 
        now = time.time()
 
        if offset is None or (now - seen) > BALL_LOST_TIMEOUT:
            # Ball lost — stop and wait for it to reappear.
            if moving:
                stop()
                moving = False
        else:
            dx, dy = offset
            dist = math.hypot(dx, dy)
 
            if dist < DEAD_ZONE_RADIUS:
                # Ball is effectively under the bot — hold position.
                if moving:
                    stop()
                    moving = False
            else:
                # Pixel offset → bearing. atan2(dx, -dy) puts "up" in the
                # image at 0°, increasing clockwise; then rotate into the
                # robot frame with CAMERA_ROTATION_OFFSET.
                bearing = math.degrees(math.atan2(dx, -dy))
                bearing = (bearing + CAMERA_ROTATION_OFFSET) % 360
                apply_action(choose_action(bearing))
                moving = True
 
        time.sleep(period)
 
    stop()
 
 
# ----------------------------------------------------------------
# Web stream + tuning UI (background thread)
# ----------------------------------------------------------------
flask_app = Flask(__name__)
 
# Single-page UI: MJPEG video on the left, one slider per detect_params key
# on the right. Sliders POST changes to /params as the user drags them.
PAGE_HTML = """
<html>
<head>
<style>
  body { margin:0; background:#111; color:#eee; font-family:sans-serif; display:flex; }
  #video { flex:1; display:flex; align-items:center; justify-content:center; background:#000; }
  #video img { max-width:100%; max-height:100vh; width:auto; height:auto; object-fit:contain; }
  #controls { width:280px; padding:16px; box-sizing:border-box; overflow-y:auto; }
  #controls label { display:block; margin-top:12px; font-size:13px; }
  #controls input[type=range] { width:100%; }
  .val { float:right; opacity:0.7; }
</style>
</head>
<body>
<div id="video"><img src="/stream"></div>
<div id="controls">
<h3>Detection tuning</h3>
<div id="sliders"></div>
</div>
 
<script>
const sliderDefs = [
  {key: 'h_low',  label: 'Hue low',  min: 0,   max: 179, step: 1},
  {key: 'h_high', label: 'Hue high', min: 0,   max: 179, step: 1},
  {key: 's_low',  label: 'Sat low',  min: 0,   max: 255, step: 1},
  {key: 's_high', label: 'Sat high', min: 0,   max: 255, step: 1},
  {key: 'v_low',  label: 'Val low',  min: 0,   max: 255, step: 1},
  {key: 'v_high', label: 'Val high', min: 0,   max: 255, step: 1},
  {key: 'min_contour_area', label: 'Min contour area', min: 0, max: 2000, step: 10},
  {key: 'centre_ignore_radius', label: 'Centre ignore radius (px)', min: 0, max: 400, step: 5},
];
 
const container = document.getElementById('sliders');
 
async function loadInitial() {
  const res = await fetch('/params');
  const current = await res.json();
 
  sliderDefs.forEach(def => {
    const wrap = document.createElement('label');
    wrap.innerHTML = `${def.label} <span class="val" id="val-${def.key}">${current[def.key]}</span>`;
 
    const input = document.createElement('input');
    input.type = 'range';
    input.min = def.min;
    input.max = def.max;
    input.step = def.step;
    input.value = current[def.key];
 
    input.addEventListener('input', () => {
      document.getElementById(`val-${def.key}`).textContent = input.value;
      fetch('/params', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({[def.key]: Number(input.value)})
      });
    });
 
    wrap.appendChild(document.createElement('br'));
    wrap.appendChild(input);
    container.appendChild(wrap);
  });
}
 
loadInitial();
</script>
</body>
</html>
"""
 
 
@flask_app.route('/')
def stream_index():
    """Serve the tuning page."""
    return PAGE_HTML
 
 
@flask_app.route('/stream')
def stream_route():
    """MJPEG stream of the latest annotated frame (multipart/x-mixed-replace)."""
    def generate():
        while isRunning:
            with stream_lock:
                jpeg = latest_jpeg
            if jpeg is not None:
                yield (
                    b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + jpeg + b'\r\n'
                )
            time.sleep(1.0 / MOTOR_LOOP_HZ)
 
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')
 
 
@flask_app.route('/params', methods=['GET', 'POST'])
def params_route():
    """GET → current detect_params as JSON. POST → merge JSON body into detect_params.
 
    Unknown keys in a POST are ignored so a stale page can't inject new fields.
    """
    if request.method == 'POST':
        update = request.get_json(force=True)
        with params_lock:
            for key, value in update.items():
                if key in detect_params:
                    detect_params[key] = value
        return jsonify({'ok': True})
 
    with params_lock:
        return jsonify(dict(detect_params))
 
 
def web_stream_loop():
    """Run the Flask server (blocking; call from a daemon thread)."""
    flask_app.run(host='0.0.0.0', port=STREAM_PORT, threaded=True, use_reloader=False)
 
 
# ----------------------------------------------------------------
# Vision
# ----------------------------------------------------------------
def find_ball(contours, centre_x, centre_y, min_area, ignore_radius):
    """Select the ball contour from a list of candidate contours.
 
    Sequence (largest contour first, first survivor wins):
      1. Reject contours smaller than `min_area` — once we hit one, every
         remaining contour is smaller too, so stop.
      2. Compute the centroid; reject if it's within `ignore_radius` px of
         image centre (own chassis, dribbler, reflections off the body).
      3. First contour to pass both is the ball.
 
    Args:
        contours:     output of cv2.findContours
        centre_x/y:   image centre in px
        min_area:     px^2 minimum contour area
        ignore_radius: px radius around centre to mask out
 
    Returns:
        (cX, cY) centroid in px, or None if nothing qualifies.
    """
    for contour in sorted(contours, key=cv2.contourArea, reverse=True):
        if cv2.contourArea(contour) <= min_area:
            break
 
        moments = cv2.moments(contour)
        if moments['m00'] == 0:
            continue
 
        cX = int(moments['m10'] / moments['m00'])
        cY = int(moments['m01'] / moments['m00'])
 
        if math.hypot(cX - centre_x, cY - centre_y) < ignore_radius:
            continue
 
        return cX, cY
 
    return None
 
 
def vision_loop():
    """Capture frames, detect the ball, publish its pixel offset.
 
    Per frame:
      1. Snapshot detect_params (so slider changes apply atomically).
      2. Capture RGB → BGR → HSV; threshold for orange.
      3. find_ball(): largest contour that clears min area AND is outside
         the centre-ignore circle.
      4. Publish ball_offset = (cX - cx, cY - cy) and last_seen.
      5. Draw debug overlay (centroid, centre→ball line, dead-zone circle,
         ignore circle), encode to JPEG for the stream, optionally show
         cv2 windows.
 
    If no acceptable contour is found, ball_offset is set to None immediately;
    motor_loop() additionally applies BALL_LOST_TIMEOUT on last_seen.
    """
    global ball_offset, last_seen, latest_jpeg
 
    picamera = Picamera2()
    picamera.configure(picamera.create_preview_configuration())
    picamera.start()
 
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, STREAM_JPEG_QUALITY]
 
    try:
        while isRunning:
            with params_lock:
                p = dict(detect_params)
 
            frame = picamera.capture_array()
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
 
            hsvFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
 
            lower_orange = np.array([p['h_low'], p['s_low'], p['v_low']])
            upper_orange = np.array([p['h_high'], p['s_high'], p['v_high']])
 
            mask = cv2.inRange(hsvFrame, lower_orange, upper_orange)
 
            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            centre_x = frame.shape[1] // 2
            centre_y = frame.shape[0] // 2
 
            ball = find_ball(
                contours, centre_x, centre_y,
                p['min_contour_area'], p['centre_ignore_radius'],
            )
 
            if ball is not None:
                cX, cY = ball
                with state_lock:
                    ball_offset = (cX - centre_x, cY - centre_y)
                    last_seen = time.time()
 
                # Debug overlay: ball centroid + centre→ball line.
                cv2.circle(frame, (cX, cY), 5, (0, 255, 0), -1)
                cv2.line(frame, (centre_x, centre_y), (cX, cY), (0, 0, 255), 2)
            else:
                with state_lock:
                    ball_offset = None
 
            # Always-on overlay: dead zone (black) and ignore radius (grey).
            cv2.circle(frame, (centre_x, centre_y), DEAD_ZONE_RADIUS, (0, 0, 0), 1)
            cv2.circle(frame, (centre_x, centre_y),
                       int(p['centre_ignore_radius']), (128, 128, 128), 1)
 
            if ENABLE_WEB_STREAM:
                ok, encoded = cv2.imencode('.jpg', frame, encode_params)
                if ok:
                    with stream_lock:
                        latest_jpeg = encoded.tobytes()
 
            if SHOW_WINDOWS:
                cv2.drawContours(frame, contours, -1, (0, 255, 0), 2)
                cv2.imshow("Frame", frame)
                cv2.imshow("Mask", mask)
                if cv2.waitKey(1) == ord('q'):
                    break
    finally:
        picamera.stop()
        if SHOW_WINDOWS:
            cv2.destroyAllWindows()
 
 
# ----------------------------------------------------------------
# Main
# ----------------------------------------------------------------
def main():
    """Bring up motors, start worker threads, run vision on the main thread.
 
    Ctrl+C (SIGINT) stops motors and exits. On any exit path the motors are
    stopped and the final slider values are printed so they can be copied
    back into detect_params as new defaults.
    """
    global isRunning
 
    setup_motors()
 
    def shutdown(sig, frame):
        global isRunning
        print("\nShutting down...")
        isRunning = False
        stop()
        sys.exit(0)
 
    signal.signal(signal.SIGINT, shutdown)
 
    motor_thread = threading.Thread(target=motor_loop, daemon=True)
    motor_thread.start()
 
    if ENABLE_WEB_STREAM:
        stream_thread = threading.Thread(target=web_stream_loop, daemon=True)
        stream_thread.start()
        print(f"Web stream at http://<pi-ip>:{STREAM_PORT}")
 
    print("Following ball. Press q in the video window (or Ctrl+C) to quit.")
 
    try:
        vision_loop()
    finally:
        isRunning = False
        motor_thread.join(timeout=1.0)
        stop()
        with params_lock:
            final_params = dict(detect_params)
        print("\nFinal detection params:")
        for key, value in final_params.items():
            print(f"  {key}: {value}")
 
 
if __name__ == "__main__":
    main()