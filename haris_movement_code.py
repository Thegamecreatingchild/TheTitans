"""
hari_movement.py — Hari's ball-pivot movement code, merged with the vision
pipeline/tuning setup from testing_rig.py, served over WebSocket instead of
Flask + MJPEG.

Mk 2 — documented + merged
12-09-2026

What changed from Hari's Mk 1
------------------------------
1. Ball detection now uses testing_rig's tuned HSV values and its find_ball()
   logic: contours are checked largest-first, and any contour whose centroid
   falls inside a "dead zone" around the image centre is rejected (that's
   the robot's own chassis/dribbler, not the ball). Hari's Mk 1 just took
   the single largest contour over a fixed area — no dead-zone exclusion.
2. Detection thresholds (HSV bounds, min contour area, dead-zone radius,
   dribble-engage radius) are no longer hardcoded / read once from
   calibration.json — they live in `ball_detect_params`, a plain dict you
   can tune live from the control page, same idea as testing_rig's sliders.
3. Added an AUTO / MANUAL mode switch and WASD(+QE) manual driving, same
   feature as testing_rig — just carried over WebSocket messages instead of
   HTTP POSTs to Flask routes, since this file's transport is already
   websockets (from Hari's version), not Flask.
4. Added the same 'v' debug hotkey: prints the current centre→ball vector
   to the console without touching the motors.
5. Dribbler is now OPTIONAL (see ENABLE_DRIBBLER below) — Hari's Mk 1 had
   no dribbler motor at all. If you don't have a 5th driver wired up yet,
   leave it off and everything else works unchanged.
6. Added motion debug logging (see DEBUG_MOTOR below): every time
   apply_auto()/apply_manual_keys() decides what to do, it prints WHY —
   which branch/key fired and the numbers behind the decision (bearing,
   distance, which key matched, etc). Rate-limited to DEBUG_PRINT_HZ lines
   per second so it stays readable even though motor_task() itself polls
   at ~50Hz. Set DEBUG_MOTOR = False to silence it once you don't need it.

What's kept from Hari's Mk 1
------------------------------
- The async/await structure (no threading — one asyncio event loop running
  camera capture, motor control, and the websocket server as concurrent
  tasks). No locks are needed the way testing_rig used them, because
  coroutines only hand control back to the loop at `await` points; as long
  as a chunk of code between awaits doesn't itself await, it's atomic.
- Hari's motor addresses (yours are different from testing_rig's, on
  purpose — untouched below).
- Hari's per-motor calibration values (SAVED_CAL).
- Hari's move(degree, speed) / spin(speed) drive functions, UNCHANGED —
  see the "Drive math" section below. Mk 2 previously replaced these with
  a generalised blend function; that's been reverted. AUTO mode now does
  exactly what Hari's Mk 1 motor_task did: spin slowly while the ball
  isn't visible, otherwise move() straight at it. Manual mode reuses the
  same two functions rather than introducing new drive math — see
  "Manual (WASD) driving" below.

Threads / tasks
----------------
This file uses asyncio tasks, not OS threads:
- stream_cam()  : camera capture + detection + websocket video broadcast
- motor_task()  : reads shared ball/key/mode state, drives the motors
- ws_handler()  : accepts one websocket connection per browser tab, and is
                  where control messages (params/mode/keys/debug) from that
                  tab are read and applied to global state
All three run concurrently under `asyncio.gather()` in main().

Hardware
--------
- 4x Steel Bar PowerfulBLDC drivers over I2C (Hari's addresses, below)
- Optional 5th driver for the dribbler, only used if ENABLE_DRIBBLER=True
- Raspberry Pi camera via picamera2
"""

import asyncio
import json
import math
import signal
import sys
import time

import cv2
import numpy as np
import board
import busio
import websockets
from picamera2 import Picamera2
from steelbar_powerful_bldc_driver import PowerfulBLDCDriver

# ----------------------------------------------------------------
# Config — hardware
# ----------------------------------------------------------------

# Hari's I2C addresses, in the order [FR, BR, BL, FL] (NOTE: this order is
# Hari's own — different from testing_rig's [FL, FR, RL, RR] order, and
# deliberately left unchanged since his physical wiring differs).
ADDRESSES = [26, 28, 27, 25]  # FR, BR, BL, FL

# Per-motor FOC calibration, index-matched to ADDRESSES above.
SAVED_CAL = [
    {'elecangleoffset': 1327731200, 'sincoscentre': 1241},  # FR
    {'elecangleoffset': 1435147520, 'sincoscentre': 1243},  # BR
    {'elecangleoffset': 1256835584, 'sincoscentre': 1258},  # BL
    {'elecangleoffset': 1150337792, 'sincoscentre': 1247},  # FL
]

MAX_SPEED = 80_000_000  # raw driver units, same ceiling Hari used

# --- Optional dribbler ---
# Hari's rig has no dribbler wired up yet. Flip this on once you add a 5th
# PowerfulBLDCDriver for it, and set DRIBBLER_ADDRESS / DRIBBLER_CAL to
# match. Every dribbler-related code path below checks this flag first,
# so leaving it False is completely safe — no 5th motor is ever touched.
ENABLE_DRIBBLER = False
DRIBBLER_ADDRESS = 29
DRIBBLER_CAL = {'elecangleoffset': 1437511680, 'sincoscentre': 1245}
DRIBBLER_SPEED = 80_000_000
dribbler_motor = None  # set in setup_motors() if ENABLE_DRIBBLER

# ----------------------------------------------------------------
# Config — networking
# ----------------------------------------------------------------
WS_PORT = 8765

# ----------------------------------------------------------------
# Config — vision / detection tuning (Nico's values, from testing_rig.py)
# ----------------------------------------------------------------

# Same HSV-threshold / geometry tunables as testing_rig's ball_detect_params,
# with the same meanings. Adjustable live from the control page — the
# capture loop reads a fresh copy of this dict every frame.
ball_detect_params = {
    'h_low': 0,
    's_low': 86,
    'v_low': 166,
    'h_high': 20,
    's_high': 255,
    'v_high': 255,
    'min_contour_area': 2,      # px^2: smallest contour accepted as the ball
    'dead_zone_radius': 135,    # px: contours whose centroid is closer than
                                 # this to image centre are ignored (own
                                 # body / dribbler / reflections)
    'ball_dribble_radius': 145, # px: distance below which we consider the
                                 # ball "possessed" and (if enabled) drive
                                 # the dribbler
}

# Degrees added to the image-derived bearing so 0deg == robot forward.
# Hari's Mk 1 instead corrected for the camera's physical mounting by
# rotating the whole frame 90deg CCW before detection (see stream_cam()).
# That rotation is kept below, so this offset defaults to 0 — only change
# it if you also change/remove the frame rotation, otherwise you'll apply
# the correction twice.
CAMERA_ROTATION_OFFSET = 0.0

# ----------------------------------------------------------------
# Shared state (read/written across the async tasks below)
# ----------------------------------------------------------------

# -- Ball state, published by stream_cam(), read by motor_task() --
ball_offset = None    # (dx, dy) px from image centre, or None if not seen
last_seen = 0.0        # time.time() of the last successful detection
BALL_LOST_TIMEOUT = 0.3  # seconds without a detection before AUTO stops

# -- Control mode: 'auto' (ball pivot) or 'manual' (browser WASD) --
control_mode = 'manual'

# -- Manual-mode held keys, posted by the browser --
active_keys = set()
keys_last_seen = 0.0
KEY_LOST_TIMEOUT = 0.3

# -- Debug: 'v' hotkey request flag, consumed by stream_cam() --
print_vector_requested = False

# -- Connected browser websocket clients (video is broadcast to all) --
clients = set()

is_running = True
motors = []  # [FR, BR, BL, FL] PowerfulBLDCDriver instances


# ----------------------------------------------------------------
# Motor setup
# ----------------------------------------------------------------
def setup_motors():
    """Initialise Hari's four drive drivers (+ optional dribbler) over I2C."""
    global motors, dribbler_motor
    i2c = busio.I2C(board.SCL, board.SDA)

    for index, address in enumerate(ADDRESSES):
        motor = PowerfulBLDCDriver(i2c, address)
        motor.set_current_limit_foc(65536 * 2)
        motor.set_id_pid_constants(1500, 200)
        motor.set_iq_pid_constants(1500, 200)
        motor.set_speed_pid_constants(4e-2, 4e-4, 3e-2)
        motor.set_ELECANGLEOFFSET(SAVED_CAL[index]["elecangleoffset"])
        motor.set_SINCOSCENTRE(SAVED_CAL[index]["sincoscentre"])
        motor.configure_operating_mode_and_sensor(3, 1)
        motor.configure_command_mode(12)
        motor.set_speed(0)
        motors.append(motor)
        print(f"Motor {index} ready")

    if ENABLE_DRIBBLER:
        dribbler_motor = PowerfulBLDCDriver(i2c, DRIBBLER_ADDRESS)
        dribbler_motor.set_current_limit_foc(65536 * 2)
        dribbler_motor.set_id_pid_constants(1500, 200)
        dribbler_motor.set_iq_pid_constants(1500, 200)
        dribbler_motor.set_speed_pid_constants(4e-2, 4e-4, 3e-2)
        dribbler_motor.set_ELECANGLEOFFSET(DRIBBLER_CAL["elecangleoffset"])
        dribbler_motor.set_SINCOSCENTRE(DRIBBLER_CAL["sincoscentre"])
        dribbler_motor.configure_operating_mode_and_sensor(3, 1)
        dribbler_motor.configure_command_mode(12)
        dribbler_motor.set_speed(0)
        print("Dribbler motor ready")

    print("All motors ready")


def set_dribbler(engaged: bool):
    """No-op unless ENABLE_DRIBBLER is True — safe to call unconditionally."""
    if not ENABLE_DRIBBLER or dribbler_motor is None:
        return
    dribbler_motor.set_speed(-DRIBBLER_SPEED if engaged else 0)


# ----------------------------------------------------------------
# Camera setup
# ----------------------------------------------------------------
# Capture resolution + target frame rate. 120fps is only actually achievable
# at a small resolution — check picam.sensor_modes on your specific camera
# and adjust CAMERA_SIZE if 120fps isn't listed for this size. Dropping
# resolution rather than fps is usually the right trade for ball tracking:
# detection only needs enough pixels to see an orange blob, not full res.
CAMERA_SIZE = (640, 480)
CAMERA_FPS = 120
CAMERA_FRAME_DURATION_US = int(1_000_000 / CAMERA_FPS)  # FrameDurationLimits is in microseconds


def setup_camera():
    """Start the Pi camera at CAMERA_SIZE / CAMERA_FPS.

    Uses create_video_configuration() instead of create_preview_configuration()
    — the preview config carries extra ISP overhead aimed at smooth on-screen
    preview, not high frame rate capture, and can silently cap fps well below
    what the sensor mode supports. FrameDurationLimits is set to the same
    min/max (both = 1/fps in microseconds) to lock the rate rather than let
    the ISP vary it with scene lighting.

    HSV thresholds now live in ball_detect_params (tunable live), so unlike
    Hari's Mk 1 this no longer reads a fixed lower/upper orange range out of
    calibration.json — only exposure/white balance/colour-gain calibration
    is loaded from there, if present.
    """
    picam = Picamera2()
    config = picam.create_video_configuration(
        main={"size": CAMERA_SIZE},
        controls={"FrameDurationLimits": (CAMERA_FRAME_DURATION_US, CAMERA_FRAME_DURATION_US)},
    )
    picam.configure(config)
    picam.start()

    try:
        with open("calibration.json") as file:
            calibration = json.load(file)
        picam.set_controls({
            "AeEnable": False,
            "AwbEnable": False,
            "ExposureTime": calibration["exposure_time"],
            "AnalogueGain": calibration["analogue_gain"],
            "ColourGains": tuple(calibration["colour_gains"]),
        })
        print(f"Loaded calibration.json (exposure/white-balance only). "
              f"Capturing at {CAMERA_SIZE[0]}x{CAMERA_SIZE[1]} @ target {CAMERA_FPS}fps.")
    except FileNotFoundError:
        print(f"No calibration.json found — using camera auto exposure/WB. "
              f"Capturing at {CAMERA_SIZE[0]}x{CAMERA_SIZE[1]} @ target {CAMERA_FPS}fps.")

    return picam


# ----------------------------------------------------------------
# Ball detection (find_ball logic ported from testing_rig.py)
# ----------------------------------------------------------------
def find_ball(contours, centre_x, centre_y, min_area, ignore_radius):
    """Pick the ball contour out of a list of HSV-mask contours.

    Same rule as testing_rig.py:
      1. Walk contours largest-first. The first one smaller than `min_area`
         means every contour after it is smaller too (sorted descending),
         so we can stop scanning right there.
      2. For each contour that clears the area check, compute its centroid
         and reject it if that centroid sits within `ignore_radius` px of
         the image centre — that's the robot's own body/dribbler/glare,
         not the ball.
      3. First contour to pass both checks wins.

    Returns (cX, cY) in pixels, or None if nothing qualifies.
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


# ----------------------------------------------------------------
# Drive math — Hari's original move()/spin(), unchanged
# ----------------------------------------------------------------
def move(degree, speed=MAX_SPEED):
    """Drive the chassis in a straight line toward `degree`.

    Exactly Hari's Mk 1 move(): each wheel is mounted at 45 degrees to the
    chassis axes, so a wheel's speed is (roughly) the dot product of the
    desired velocity vector with that wheel's roller axis. Rotating the
    requested direction by 45 degrees and reading off cos/sin gives the
    FR/BR wheel components directly; the other two wheels are the mirror
    image (hence the sign flips), since they're mounted on the opposite
    side of the chassis.

        angle = radians(degree + 45)
        x = cos(angle) * speed
        y = sin(angle) * speed

    Args:
        degree: direction to drive in, 0 == robot forward, increasing
                clockwise (same convention as ball_angle below).
        speed:  raw driver units.

    Wheel order matches Hari's ADDRESSES/motors list: motors[0]=FR,
    [1]=BR, [2]=BL, [3]=FL.
    """
    angle_rad = math.radians(degree + 45)
    x = math.floor(math.cos(angle_rad) * speed)
    y = math.floor(math.sin(angle_rad) * speed)
    motors[0].set_speed(y + x)      # FR
    motors[1].set_speed(y - x)      # BR
    motors[2].set_speed(-(y + x))   # BL
    motors[3].set_speed(-(y - x))   # FL


def spin(speed):
    """Turn every wheel the same direction/speed — spins the chassis on the
    spot (same as Hari's Mk 1 spin()). Positive = the direction Hari used
    while searching for the ball.
    """
    for motor in motors:
        motor.set_speed(speed)


def stop():
    for motor in motors:
        motor.set_speed(0)
    set_dribbler(False)


# ----------------------------------------------------------------
# Motion debug logging
# ----------------------------------------------------------------
# When True, apply_manual_keys() / apply_auto() print a line each time they
# run explaining WHY the motors are being told to do what they're being
# told to do — which key(s)/branch fired, and the numbers that drove the
# decision. Throttled below so it's readable instead of a 50Hz flood.
#
# This is the STARTUP default. It can also be flipped at runtime without
# restarting the script by sending {"type": "debug_motor", "enabled": bool}
# over the websocket (see ws_handler()) — e.g. a toggle button/hotkey on
# the control page.
DEBUG_MOTOR = False

# Console output is printed at most this many times per second (motor_task
# itself still runs at ~50Hz — this only limits how often we print).
DEBUG_PRINT_HZ = 5
_debug_last_print = 0.0


def _debug(message):
    """Print `message` if DEBUG_MOTOR is on, rate-limited to DEBUG_PRINT_HZ."""
    global _debug_last_print
    if not DEBUG_MOTOR:
        return
    now = time.time()
    if now - _debug_last_print < (1.0 / DEBUG_PRINT_HZ):
        return
    _debug_last_print = now
    print(message)


# ----------------------------------------------------------------
# Manual (WASD) driving — built from Hari's move()/spin(), not a new
# drive model. Each key just calls move() at a fixed compass degree, or
# spin() for rotation, same as Hari's functions are meant to be driven.
# Unlike testing_rig's key blending, held keys aren't summed into a
# combined vector — if more than one translate/rotate key is held, the
# first match below wins, keeping this a thin wrapper around move()/spin()
# rather than new math layered on top.
# ----------------------------------------------------------------
KEY_DEGREES = {
    'w': 0,      # forward
    'd': 90,     # strafe right
    's': 180,    # back
    'a': 270,    # strafe left
}
ROTATE_KEYS = {
    'q': -1,     # spin CCW
    'e': 1,      # spin CW
}
DRIBBLE_KEY = 'k'


def apply_manual_keys(keys):
    """Translate OR rotate from the held key set, via move()/spin() directly."""
    for key, sign in ROTATE_KEYS.items():
        if key in keys:
            speed = sign * int(MAX_SPEED * 0.3)
            _debug(f"[manual] held={sorted(keys)} -> rotate key '{key}' matched first "
                   f"-> spin(speed={speed}) ({'CW' if sign > 0 else 'CCW'})")
            spin(speed)
            set_dribbler(ENABLE_DRIBBLER and DRIBBLE_KEY in keys)
            return

    for key, degree in KEY_DEGREES.items():
        if key in keys:
            speed = int(MAX_SPEED * 0.7)
            _debug(f"[manual] held={sorted(keys)} -> translate key '{key}' matched first "
                   f"-> move(degree={degree}, speed={speed})")
            move(degree, speed)
            set_dribbler(ENABLE_DRIBBLER and DRIBBLE_KEY in keys)
            return

    _debug(f"[manual] held={sorted(keys)} -> no recognised key held -> stop()")
    stop()


# ----------------------------------------------------------------
# AUTO mode — exactly Hari's Mk 1 motor_task behaviour
# ----------------------------------------------------------------
def apply_auto(ball_visible, ball_angle, distance):
    """Unchanged from Hari's Mk 1: spin to search while the ball isn't
    visible, otherwise drive straight at it via move(ball_angle, ...).
    No heading-error correction, no proportional turning — the ball's
    angle from image centre IS the drive direction, same as Mk 1.
    """
    if not ball_visible:
        speed = int(MAX_SPEED * 0.3)
        _debug(f"[auto] ball not visible (or stale) -> spin(speed={speed}) to search")
        spin(speed)
        set_dribbler(False)
        return

    speed = int(MAX_SPEED * 0.7)
    _debug(f"[auto] ball at angle={ball_angle:.1f}deg, distance={distance:.1f}px "
           f"-> move(degree={ball_angle:.1f}, speed={speed})")
    move(ball_angle, speed)

    if distance <= ball_detect_params['ball_dribble_radius']:
        _debug(f"[auto]   distance {distance:.1f}px <= dribble radius "
               f"{ball_detect_params['ball_dribble_radius']}px -> dribbler ENGAGE"
               f"{'' if ENABLE_DRIBBLER else ' (skipped, ENABLE_DRIBBLER=False)'}")
        set_dribbler(True)
    else:
        set_dribbler(False)


# ----------------------------------------------------------------
# Async tasks
# ----------------------------------------------------------------
async def ws_handler(ws):
    """One instance per connected browser tab.

    Broadcasts video frames to everyone (added to `clients`), and reads
    control messages from THIS tab specifically:
      {"type": "params", ...}         -> merge into ball_detect_params
      {"type": "mode", "mode": ...}   -> set control_mode ('auto'/'manual')
      {"type": "keys", "keys": [...]} -> replace active_keys, refresh timeout
      {"type": "debug_vector"}        -> request a one-off console printout
      {"type": "debug_motor", "enabled": bool} -> toggle DEBUG_MOTOR on/off

    This mirrors testing_rig's /params, /mode, /keys, /debug/print_vector
    Flask routes, just carried as JSON text frames on the same websocket
    connection instead of separate HTTP endpoints.
    """
    global control_mode, keys_last_seen, print_vector_requested, DEBUG_MOTOR

    clients.add(ws)
    print("Browser connected")
    try:
        async for message in ws:
            # Video frames are sent by the server as bytes; anything a
            # client sends us is a control message, always JSON text.
            if not isinstance(message, str):
                continue

            try:
                data = json.loads(message)
            except json.JSONDecodeError:
                continue

            msg_type = data.get('type')

            if msg_type == 'params':
                for key, value in data.items():
                    if key in ball_detect_params and key != 'type':
                        ball_detect_params[key] = value

            elif msg_type == 'mode':
                new_mode = data.get('mode')
                if new_mode in ('auto', 'manual'):
                    control_mode = new_mode
                    if new_mode != 'manual':
                        active_keys.clear()

            elif msg_type == 'keys':
                incoming = data.get('keys', [])
                valid = {k for k in incoming
                         if k in KEY_DEGREES or k in ROTATE_KEYS or k == DRIBBLE_KEY}
                active_keys.clear()
                active_keys.update(valid)
                keys_last_seen = time.time()

            elif msg_type == 'debug_vector':
                print_vector_requested = True

            elif msg_type == 'debug_motor':
                enabled = bool(data.get('enabled', not DEBUG_MOTOR))
                DEBUG_MOTOR = enabled
                print(f"[debug] motor debug logging {'ON' if enabled else 'OFF'}")
    finally:
        clients.discard(ws)
        print("Browser disconnected")


async def stream_cam(picam):
    """Capture, detect, publish ball_offset/last_seen, broadcast video.

    Per frame:
      1. Snapshot ball_detect_params (so a slider change mid-frame doesn't
         apply half-updated — same reasoning as testing_rig's params_lock,
         just without needing an actual lock since this is all synchronous
         code between awaits).
      2. Rotate the frame 90deg CCW — Hari's fix for the camera's physical
         mounting angle (kept from Mk 1; see CAMERA_ROTATION_OFFSET above
         for why the bearing offset is 0 while this stays in place).
      3. Threshold HSV for the ball colour, find contours, hand them to
         find_ball() (testing_rig's dead-zone-aware selection).
      4. Publish ball_offset = (dx, dy) from frame centre + last_seen.
      5. Draw the same debug overlay testing_rig drew: ball centroid,
         centre->ball line, dead-zone circle.
      6. If 'v' was pressed on the page, print the vector once.
      7. Encode to JPEG and broadcast to every connected browser tab.
    """
    global ball_offset, last_seen, print_vector_requested

    while is_running:
        with_params = dict(ball_detect_params)

        frame = picam.capture_array()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

        height, width = frame.shape[:2]
        centre_x, centre_y = width // 2, height // 2

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower = np.array([with_params['h_low'], with_params['s_low'], with_params['v_low']])
        upper = np.array([with_params['h_high'], with_params['s_high'], with_params['v_high']])
        mask = cv2.inRange(hsv, lower, upper)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        ball = find_ball(
            contours, centre_x, centre_y,
            with_params['min_contour_area'], with_params['dead_zone_radius'],
        )

        cv2.circle(frame, (centre_x, centre_y), 4, (0, 255, 0), -1)
        cv2.circle(frame, (centre_x, centre_y),
                   int(with_params['dead_zone_radius']), (128, 128, 128), 2)

        if ball is not None:
            cX, cY = ball
            ball_offset = (cX - centre_x, cY - centre_y)
            last_seen = time.time()
            cv2.circle(frame, (cX, cY), 5, (0, 0, 255), -1)
            cv2.line(frame, (centre_x, centre_y), (cX, cY), (255, 0, 0), 2)
        else:
            ball_offset = None

        if print_vector_requested:
            print_vector_requested = False
            if ball_offset is None:
                print("[v] No ball currently detected — no vector to print.")
            else:
                dx, dy = ball_offset
                distance = math.hypot(dx, dy)
                bearing = math.degrees(math.atan2(dx, -dy)) % 360
                print(f"[v] Ball vector: dx={dx}px, dy={dy}px, "
                      f"distance={distance:.1f}px, bearing={bearing:.1f}deg")

        success, encoded = cv2.imencode('.jpg', frame)
        if success and clients:
            jpg = encoded.tobytes()
            await asyncio.gather(*[c.send(jpg) for c in clients], return_exceptions=True)

        await asyncio.sleep(0.03)


async def motor_task():
    """Read shared mode/ball/key state and drive the motors accordingly.

    AUTO:   ball not seen for > BALL_LOST_TIMEOUT -> stop
            otherwise                              -> apply_auto()
    MANUAL: no keys held, or stale (> KEY_LOST_TIMEOUT) -> stop
            otherwise                                    -> apply_manual_keys()
    """
    while is_running:
        now = time.time()

        if control_mode == 'manual':
            if not active_keys or (now - keys_last_seen) > KEY_LOST_TIMEOUT:
                _debug("[manual] no keys held or stale (> KEY_LOST_TIMEOUT) -> stop()")
                stop()
            else:
                apply_manual_keys(active_keys)

        else:  # 'auto'
            stale = ball_offset is None or (now - last_seen) > BALL_LOST_TIMEOUT
            ball_visible = not stale
            if ball_visible:
                dx, dy = ball_offset
                distance = math.hypot(dx, dy)
                bearing = math.degrees(math.atan2(dx, -dy)) % 360
                bearing = (bearing + CAMERA_ROTATION_OFFSET) % 360
                apply_auto(True, bearing, distance)
            else:
                if ball_offset is None:
                    _debug("[auto] no ball detected this frame")
                else:
                    _debug(f"[auto] last detection {now - last_seen:.2f}s ago "
                           f"> BALL_LOST_TIMEOUT ({BALL_LOST_TIMEOUT}s) -> treated as not visible")
                apply_auto(False, 0.0, 0.0)

        await asyncio.sleep(0.02)


# ----------------------------------------------------------------
# Main
# ----------------------------------------------------------------
async def main():
    setup_motors()
    picam = setup_camera()

    server = await websockets.serve(ws_handler, "0.0.0.0", WS_PORT)
    print(f"WebSocket control + video on port {WS_PORT} — open control.html to drive.")
    print("Ctrl+C to stop.")

    await asyncio.gather(
        stream_cam(picam),
        motor_task(),
    )
    await server.wait_closed()


def shutdown(sig, frame):
    global is_running
    print("\nShutting down...")
    is_running = False
    stop()
    sys.exit(0)


if __name__ == "__main__":
    signal.signal(signal.SIGINT, shutdown)
    asyncio.run(main())