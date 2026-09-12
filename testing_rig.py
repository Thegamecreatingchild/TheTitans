"""
testing_rig.py — Testing ground for a MHS robocupjunior bot.

V1.4
What it does rn
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
5. NEW: the web page has an AUTO / MANUAL switch. In MANUAL mode the page
   captures WASD(+QE) keydown/keyup events in the browser and posts the held
   key set to the Pi, so you get the old WASD.py teleop behaviour but driven
   from the same page as the stream/sliders instead of a curses terminal.
   In AUTO mode the bot follows the ball exactly as before.
6. NEW: pressing 'v' on the page prints the current ball vector (the offset
   from image centre that the centre→ball debug line represents) to the
   Pi's console. Works in either mode and doesn't affect the motors.

Threads
-------
- main thread      : vision_loop()   — camera capture + detection (main thread so
                                       cv2.imshow works when SHOW_WINDOWS=True)
- motor_thread     : motor_loop()    — reads shared mode + ball/key state, commands motors
- stream_thread    : web_stream_loop() — Flask MJPEG + slider + mode + key endpoints

Shared state is protected by five locks: state_lock (ball position + mode),
params_lock (detection thresholds), stream_lock (latest JPEG), keys_lock
(manual-mode held keys), debug_lock (print-vector request flag).

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
MOTOR_ADDRESSES = [25, 28, 27, 26, 29]

# Per-motor FOC calibration captured from a previous calibration run.
# Order matches MOTOR_ADDRESSES. Re-run calibration if a motor or driver is swapped.
SAVED_CAL = [
    {'elecangleoffset': 1327731200, 'sincoscentre': 1241},  # FL
    {'elecangleoffset': 1435147520, 'sincoscentre': 1243},  # FR
    {'elecangleoffset': 1256835584, 'sincoscentre': 1258},  # RL
    {'elecangleoffset': 1150337792, 'sincoscentre': 1247},  # RR
    {'elecangleoffset': 1437511680, 'sincoscentre': 1245},  # dribbler
]

# Base speed command sent to the drivers (raw driver units, not RPM).
# Actual per-motor command = pattern_sign * SPEED * MOTOR_SPEED_SCALE[motor].
SPEED = 120_000_000

# Per-motor multiplier applied to SPEED. Use this to trim motors that spin
# faster/slower than the others so the bot tracks straight. This is a
# hardware trim, so it's shared by both AUTO and MANUAL driving.
MOTOR_SPEED_SCALE = {
    'FL': 1.0,
    'FR': 0.50,
    'RL': 0.50,
    'RR': 0.50,
    'dribbler': 1.0,
}

# Motor name → index into `motors` / MOTOR_ADDRESSES.
MOTOR_INDEX = {
    'FL': 0,
    'FR': 1,
    'RL': 2,
    'RR': 3,
    'dribbler': 4
}

# Direction sign for each motor per AUTO action. Signs encode both the
# mechanical mounting of each motor (left side is mirrored) and the wheel
# geometry. 'back' is kept for manual testing but is NOT used by the follow
# logic — the bot rotates to face the ball instead of reversing.
MOTOR_ACTIONS = {
    'forward': {'FL': -1, 'FR': 1,  'RL': -1, 'RR': 1},
    'back':    {'FL': 1,  'FR': -1, 'RL': 1,  'RR': -1},
    'cw':      {'FL': -1, 'FR': -1, 'RL': -1, 'RR': -1},
    'ccw':     {'FL': 1,  'FR': 1,  'RL': 1,  'RR': 1},
    'dribble': {'dribbler': -1},
    'shoot': {'dribbler': 1}
}

# Key -> per-motor multiplier for MANUAL mode, taken from WASD.py. Contributions
# from held keys are summed per motor (then clamped to [-1, 1]) so combos like
# w+d blend naturally.
#   w / s  -> straight translation, forward / back
#   a      -> strafe
#   d / e  -> rotate CW
#   q      -> rotate CCW
# See WASD.py for the full explanation of why translation uses the
# side-split pattern and rotation uses the uniform pattern on this hardware.
KEY_ACTIONS = {
    'w': {'FL': -1, 'FR': 1,  'RL': -1, 'RR': 1},   # forward
    's': {'FL': 1,  'FR': -1, 'RL': 1,  'RR': -1},  # back
    'a': {'FL': 1,  'FR': 1,  'RL': -1, 'RR': -1},  # strafe
    'd': {'FL': -1, 'FR': -1, 'RL': 1,  'RR': 1},   # rotate CW
    'e': {'FL': -1, 'FR': -1, 'RL': -1, 'RR': -1},  # rotate CCW
    'q': {'FL': 1,  'FR': 1,  'RL': 1,  'RR': 1},   # rotate CW
    'k': {'dribbler': -1},   # dribbler forward
    'l': {'dribbler': 1},  # dribbler back
}

# Degrees added to the image-derived bearing so that 0° == robot forward.
# Accounts for the camera being mounted rotated relative to the chassis.
CAMERA_ROTATION_OFFSET = -90.0

# ================================================================
# TUNE-ME BLOCK - everything you'll adjust during testing lives here
# ================================================================

# Bearing error (degrees) within which the bot drives forward instead of
# rotating. Larger = drives sooner but tracks less precisely.
FORWARD_ALIGN_TOLERANCE = 15.0

# Seconds without a detection before the bot stops (AUTO mode). Bridges
# single-frame dropouts without letting the bot keep driving on stale data.
BALL_LOST_TIMEOUT = 0.3

# Seconds without a key update before the bot stops (MANUAL mode). Guards
# against a dropped/late browser request leaving the bot driving blind.
KEY_LOST_TIMEOUT = 0.3

# Motor control loop rate.
MOTOR_LOOP_HZ = 50

# Open local cv2 windows (needs a desktop session). Off for headless runs.
SHOW_WINDOWS = False

# Serve the MJPEG stream + tuning sliders at http://<pi-ip>:STREAM_PORT
ENABLE_WEB_STREAM = True
STREAM_PORT = 5000
STREAM_JPEG_QUALITY = 80

def pixel_dist_to_cm(pixel_distance):
    return 260 - (2400/pixel_distance+11)

# ================================================================

# ----------------------------------------------------------------
# Tunable detection parameters, adjustable live via sliders on the
# web page. vision_loop() reads a snapshot of this each frame -
# the sliders just write into it. Defaults match the last hardcoded
# values you were using.
# ----------------------------------------------------------------
params_lock = threading.Lock()
ball_detect_params = {
    # HSV threshold bounds for the orange ball (OpenCV ranges: H 0-179, S/V 0-255).
    'h_low': 0,
    's_low': 86,
    'v_low': 166,
    'h_high': 20,
    's_high': 255,
    'v_high': 255,
    'min_contour_area': 2,       # px^2: smallest contour accepted as the ball
    'dead_zone_radius': 135,   # px: contours whose centroid is closer than
                                  # this to image centre are ignored (own body,
                                  # dribbler, reflections). Must be < DEAD_ZONE_RADIUS
                                  # or the bot can never "arrive" at the ball.
    'ball_dribble_radius': 145
}

cyan_goal_detect_params = {
    'h_low': 80,
    's_low': 100,
    'v_low': 36,
    'h_high': 125,
    's_high': 255,
    'v_high': 255,
    'min_contour_area': 120
}

yellow_goal_detect_params = {
    'h_low': 22,
    's_low': 86,
    'v_low': 100,
    'h_high': 40,
    's_high': 255,
    'v_high': 255,
    'min_contour_area': 120
}

# --- Shared ball state (vision writes, motor loop reads) ---
state_lock = threading.Lock()
ball_offset = None   # (dx, dy) pixels from image centre, or None if not seen
last_seen = 0.0      # time.time() of the last successful detection
is_running : bool = True     # global run flag; set False to stop all loops
last_known_position : tuple = (None, None)

# -- Goal scoring stuff (possesion state, dribbler position offset)
has_posession : bool = False  # True if the ball is within DEAD_ZONE_RADIUS of centre
last_known_position_offset : float = 10.5
is_scoring_to_cyan_goal : bool = False

# --- Mode switch: 'auto' (ball tracking) or 'manual' (browser WASD) ---
mode_lock = threading.Lock()
control_mode = 'manual'

# --- Manual-mode held-key state, posted by the browser (see /keys route) ---
keys_lock = threading.Lock()
active_keys = set()
keys_last_seen = 0.0

motors = []          # list[PowerfulBLDCDriver], populated by setup_motors()

# --- Latest encoded frame for the MJPEG stream ---
stream_lock = threading.Lock()
latest_jpeg = None

# --- Debug: "print the current ball vector" request flag. Set by the 'v'
# key on the web page (via /debug/print_vector), consumed and cleared by
# vision_loop() on the next frame it processes. This is deliberately kept
# separate from KEY_ACTIONS/active_keys so it never affects motor commands.
debug_lock = threading.Lock()
print_vector_requested = False


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

        motor.set_current_limit_foc(65536 * 2)
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
    """Commands all motors to do a specific action"""
    pattern = MOTOR_ACTIONS[action_name]
    for motor_name, value in pattern.items():
        index = MOTOR_INDEX[motor_name]
        speed = int(value * SPEED * MOTOR_SPEED_SCALE[motor_name])
        motors[index].set_speed(speed)


def apply_keys(keys):
    """Command all four motors from a set of held WASD(+QE) keys (MANUAL mode).

    Sums each held key's per-motor contribution from KEY_ACTIONS, clamps each
    motor's total to [-1, 1], then scales by SPEED and that motor's
    MOTOR_SPEED_SCALE entry — same blending logic as WASD.py's move().
    """
    totals = {name: 0.0 for name in MOTOR_INDEX}

    for key in keys:
        action = KEY_ACTIONS.get(key)
        if action is None:
            continue
        for motor_name, value in action.items():
            totals[motor_name] += value

    for motor_name, index in MOTOR_INDEX.items():
        clamped = max(-1.0, min(1.0, totals[motor_name]))
        speed = int(clamped * SPEED * MOTOR_SPEED_SCALE[motor_name])
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
    """Poll the shared mode + ball/key state at MOTOR_LOOP_HZ and drive the motors.

    AUTO mode behaviour:
      - ball not seen for > BALL_LOST_TIMEOUT  → stop
      - ball inside DEAD_ZONE_RADIUS           → stop (arrived)
      - otherwise                              → rotate to face / drive forward

    MANUAL mode behaviour:
      - no keys held, or no update for > KEY_LOST_TIMEOUT → stop
      - otherwise                                          → apply_keys()

    `moving` tracks whether a stop() has already been issued so we don't
    hammer the I2C bus with redundant zero-speed commands. Switching mode
    forces one stop() so nothing carries over from the old mode.
    """
    period = 1.0 / MOTOR_LOOP_HZ
    moving = False
    last_mode = None

    while is_running:
        with mode_lock:
            mode = control_mode

        if mode != last_mode:
            # Mode just changed — always stop once so we don't inherit
            # leftover motion from the previous mode.
            stop()
            moving = False
            last_mode = mode

        now = time.time()

        if mode == 'manual':
            with keys_lock:
                keys = set(active_keys)
                seen = keys_last_seen

            if not keys or (now - seen) > KEY_LOST_TIMEOUT:
                if moving:
                    stop()
                    moving = False
            else:
                apply_keys(keys)
                moving = True

        else:  # 'auto'
            with state_lock:
                offset = ball_offset
                seen = last_seen

            if offset is None or (now - seen) > BALL_LOST_TIMEOUT:
                if moving:
                    stop()
                    moving = False
            else:
                dx, dy = offset
                distance = math.hypot(dx, dy)
                
                bearing = math.degrees(math.atan2(dx, -dy))
                bearing = (bearing + CAMERA_ROTATION_OFFSET) % 360
                apply_action(choose_action(bearing))

                if distance <= ball_detect_params["ball_dribble_radius"]:
                    apply_action('dribble')
                else:
                    motors[MOTOR_INDEX['dribbler']].set_speed(0)

                moving = True

        time.sleep(period)

    stop()


# ----------------------------------------------------------------
# Web stream + tuning UI (background thread)
# ----------------------------------------------------------------
flask_app = Flask(__name__)

# Single-page UI: MJPEG video on the left, mode switch + one slider per
# ball_detect_params key on the right. Sliders POST changes to /params as the
# user drags them. The AUTO/MANUAL switch POSTs to /mode. In MANUAL mode,
# WASD(+QE) keydown/keyup events are captured and POSTed to /keys. Pressing
# 'v' (either mode) POSTs to /debug/print_vector, which asks the Pi console
# to print the current ball vector on the next processed frame.
PAGE_HTML = """
<html>
<head>
<style>
  body { margin:0; background:#111; color:#eee; font-family:sans-serif; display:flex; }
  #video { flex:1; display:flex; align-items:center; justify-content:center; background:#000; }
  #video img { max-width:100%; max-height:100vh; width:auto; height:auto; object-fit:contain; }
  #controls { width:300px; padding:16px; box-sizing:border-box; overflow-y:auto; }
  #controls label { display:block; margin-top:12px; font-size:13px; }
  #controls input[type=range] { width:100%; }
  .val { float:right; opacity:0.7; }

  #modeSwitch { display:flex; border:1px solid #444; border-radius:6px; overflow:hidden; margin-bottom:8px; }
  #modeSwitch button { flex:1; padding:10px; border:none; background:#222; color:#eee; font-size:14px; cursor:pointer; }
  #modeSwitch button.active { background:#3a7; color:#111; font-weight:bold; }

  #manualPanel { display:none; margin-top:12px; padding:12px; border:1px solid #444; border-radius:6px; }
  #manualPanel.visible { display:block; }
  #keyGrid { display:grid; grid-template-columns:repeat(3, 40px); gap:6px; justify-content:center; margin-top:8px; }
  .keycap { width:40px; height:40px; display:flex; align-items:center; justify-content:center;
            border:1px solid #555; border-radius:4px; background:#222; font-size:13px; }
  .keycap.held { background:#3a7; color:#111; border-color:#3a7; }
</style>
</head>
<body>
<div id="video"><img src="/stream"></div>
<div id="controls">

<h3>Control mode</h3>
<div id="modeSwitch">
  <button id="btnAuto">AUTO (ball)</button>
  <button id="btnManual">MANUAL (WASD)</button>
</div>

<div id="manualPanel">
  Click the video area first so your keystrokes are captured, then hold
  W/A/S/D to translate and Q/E (or D) to rotate. Releasing all keys stops
  the bot. Keys stop being sent if this tab loses focus.
  <div id="keyGrid">
    <div></div><div class="keycap" data-key="w">W</div><div></div>
    <div class="keycap" data-key="a">A</div><div class="keycap" data-key="s">S</div><div class="keycap" data-key="d">D</div>
    <div class="keycap" data-key="q">Q</div><div></div><div class="keycap" data-key="e">E</div>
    <div class="keycap" data-key="k">K</div><div></div><div class="keycap" data-key="l">L</div>
  </div>
</div>

<h3>Debug</h3>
<div style="font-size:13px; opacity:0.8;">
  Press <b>V</b> (anywhere, either mode) to print the current ball vector
  (centre→ball offset in px) to the Pi's console.
</div>

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
  {key: 'dead_zone_radius', label: 'Dead zone radius (px)', min: 0, max: 400, step: 5},
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

// ---- Mode switch ----
const btnAuto = document.getElementById('btnAuto');
const btnManual = document.getElementById('btnManual');
const manualPanel = document.getElementById('manualPanel');
let currentMode = 'auto';

function renderMode() {
  btnAuto.classList.toggle('active', currentMode === 'auto');
  btnManual.classList.toggle('active', currentMode === 'manual');
  manualPanel.classList.toggle('visible', currentMode === 'manual');
  if (currentMode !== 'manual') {
    clearKeys();
  }
}

async function setMode(mode) {
  currentMode = mode;
  renderMode();
  await fetch('/mode', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({mode})
  });
}

btnAuto.addEventListener('click', () => setMode('auto'));
btnManual.addEventListener('click', () => setMode('manual'));

async function loadInitialMode() {
  const res = await fetch('/mode');
  const data = await res.json();
  currentMode = data.mode;
  renderMode();
}
loadInitialMode();

// ---- Manual WASD key capture ----
const VALID_KEYS = ['w', 'a', 's', 'd', 'q', 'e', 'k', 'l'];
const pressedKeys = new Set();
let sendTimer = null;

function keycapFor(key) {
  return document.querySelector(`.keycap[data-key="${key}"]`);
}

function sendKeys() {
  fetch('/keys', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({keys: Array.from(pressedKeys)})
  });
}

function clearKeys() {
  pressedKeys.forEach(k => { const cap = keycapFor(k); if (cap) cap.classList.remove('held'); });
  pressedKeys.clear();
  sendKeys();
}

window.addEventListener('keydown', (e) => {
  const key = e.key.toLowerCase();

  // 'v' is a standalone debug request, valid in either mode, and does not
  // go through the held-keys/motor pipeline at all.
  if (key === 'v') {
    e.preventDefault();
    fetch('/debug/print_vector', { method: 'POST' });
    return;
  }

  if (currentMode !== 'manual') return;
  if (!VALID_KEYS.includes(key)) return;
  e.preventDefault();
  if (!pressedKeys.has(key)) {
    pressedKeys.add(key);
    const cap = keycapFor(key);
    if (cap) cap.classList.add('held');
    sendKeys();
  }
});

window.addEventListener('keyup', (e) => {
  if (currentMode !== 'manual') return;
  const key = e.key.toLowerCase();
  if (!VALID_KEYS.includes(key)) return;
  e.preventDefault();
  if (pressedKeys.has(key)) {
    pressedKeys.delete(key);
    const cap = keycapFor(key);
    if (cap) cap.classList.remove('held');
    sendKeys();
  }
});

// Safety: stop driving if the tab loses focus/visibility.
window.addEventListener('blur', clearKeys);
document.addEventListener('visibilitychange', () => { if (document.hidden) clearKeys(); });

// Heartbeat: re-send the held key set periodically so a single dropped
// request (or a very low browser keyboard-repeat rate) doesn't trip
// the server-side KEY_LOST_TIMEOUT while keys are genuinely still held.
setInterval(() => {
  if (currentMode === 'manual' && pressedKeys.size > 0) {
    sendKeys();
  }
}, 100);
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
        while is_running:
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
    """GET → current ball_detect_params as JSON. POST → merge JSON body into ball_detect_params.

    Unknown keys in a POST are ignored so a stale page can't inject new fields.
    """
    if request.method == 'POST':
        update = request.get_json(force=True)
        with params_lock:
            for key, value in update.items():
                if key in ball_detect_params:
                    ball_detect_params[key] = value
        return jsonify({'ok': True})

    with params_lock:
        return jsonify(dict(ball_detect_params))


@flask_app.route('/mode', methods=['GET', 'POST'])
def mode_route():
    """GET → current control_mode as JSON. POST → set control_mode.

    Only 'auto' and 'manual' are accepted; anything else is ignored (and the
    current mode is returned) so a stale/broken client can't wedge the bot
    into an unknown state.
    """
    global control_mode

    if request.method == 'POST':
        body = request.get_json(force=True)
        new_mode = body.get('mode')
        if new_mode in ('auto', 'manual'):
            with mode_lock:
                control_mode = new_mode
            # Switching away from manual should immediately drop any held
            # keys server-side too, in case the browser's clearKeys() call
            # doesn't arrive (e.g. connection hiccup).
            if new_mode != 'manual':
                with keys_lock:
                    active_keys.clear()
        with mode_lock:
            return jsonify({'mode': control_mode})

    with mode_lock:
        return jsonify({'mode': control_mode})


@flask_app.route('/keys', methods=['POST'])
def keys_route():
    """POST → replace the held-key set for MANUAL mode.

    Body: {"keys": ["w", "d", ...]}. Unknown keys are dropped. Updates
    keys_last_seen so motor_loop's KEY_LOST_TIMEOUT check knows this is fresh.
    """
    global keys_last_seen

    body = request.get_json(force=True)
    incoming = body.get('keys', [])
    valid = {k for k in incoming if k in KEY_ACTIONS}

    with keys_lock:
        active_keys.clear()
        active_keys.update(valid)
        keys_last_seen = time.time()

    return jsonify({'ok': True, 'keys': sorted(valid)})


@flask_app.route('/debug/print_vector', methods=['POST'])
def debug_print_vector_route():
    """POST → ask vision_loop() to print the current ball vector once.

    Just flips a flag; vision_loop() does the actual printing (and clears
    the flag) the next time it processes a frame, so the printed value is
    always the freshest one available and this stays a no-op on the motors.
    """
    global print_vector_requested

    with debug_lock:
        print_vector_requested = True

    return jsonify({'ok': True})


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

        last_known_position = (cX, cY)
        return cX, cY
    return None


def find_goal(contours, centre_x, centre_y, min_area):
    """Select the goal contour from a list of candidate contours.

    Sequence (largest contour first, first survivor wins):
      1. Reject contours smaller than `min_area` — once we hit one, every
         remaining contour is smaller too, so stop.
      2. Compute the centroid; reject if it's within `ignore_radius` px of
         image centre (own chassis, dribbler, reflections off the body).
      3. First contour to pass both is the goal.

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

        last_known_position = (cX, cY)
        return cX, cY
    return None

def vision_loop():
    """Capture frames, detect the ball, publish its pixel offset.

    Per frame:
      1. Snapshot ball_detect_params (so slider changes apply atomically).
      2. Capture RGB → BGR → HSV; threshold for orange.
      3. find_ball(): largest contour that clears min area AND is outside
         the centre-ignore circle.
      4. Publish ball_offset = (cX - cx, cY - cy) and last_seen.
      5. Draw debug overlay (centroid, centre→ball line, dead-zone circle,
         ignore circle), encode to JPEG for the stream, optionally show
         cv2 windows.
      6. If the 'v' key was pressed on the web page since the last frame,
         print the centre→ball vector (the same one drawn as the debug
         line) to the console and clear the request.

    This runs regardless of control_mode — the stream and detection stay
    live in MANUAL mode too, so you can still see the ball while driving
    by hand; motor_loop() is what decides whether to act on it.

    If no acceptable contour is found, ball_offset is set to None immediately;
    motor_loop() additionally applies BALL_LOST_TIMEOUT on last_seen.
    """
    global ball_offset, last_seen, latest_jpeg, print_vector_requested

    picamera = Picamera2()
    picamera.configure(picamera.create_preview_configuration())
    picamera.start()

    encode_params = [cv2.IMWRITE_JPEG_QUALITY, STREAM_JPEG_QUALITY]

    try:
        while is_running:
            with params_lock:
                p = dict(ball_detect_params)
                g = dict(cyan_goal_detect_params) if is_scoring_to_cyan_goal else dict(yellow_goal_detect_params)

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
                p['min_contour_area'], p['dead_zone_radius'],
            )

            goal = find_goal(
                contours, centre_x, centre_y,
                g['min_contour_area'],
            )

            if ball is not None and has_posession == False:
                cX, cY = ball
                with state_lock:
                    ball_offset = (cX - centre_x, cY - centre_y)
                    last_seen = time.time()

                # Debug overlay: ball centroid + centre→ball line.
                cv2.circle(frame, (cX, cY), 5, (0, 255, 0), -1)
                cv2.line(frame, (centre_x, centre_y), (cX, cY), (0, 0, 255), 2)
                
            # elif last_known_position[0] < last_known_position_offset or has_possesion is True:
            #     cX, cY = goal
            #     has_posession = True
            #     with state_lock:
            #         goal_offset = (cX - centre_x, cY - centre_y)
            #         last_seen = time.time()
                
            #     cv2.circle(frame, (cX, cY), 5, (0, 255, 0), -1)
            #     cv2.line(frame, (centre_x, centre_y), (cX, cY), (0, 0, 255), 2)

            else:
                with state_lock:
                    ball_offset = None

            # Always-on overlay: dead zone (black) and ignore radius (grey).
            cv2.circle(frame, (centre_x, centre_y), int(p['dead_zone_radius']), (128, 128, 128), 10)            
            
            # Debug: print the current centre→ball vector if 'v' was pressed
            # on the web page since the last frame. Check-and-clear under
            # debug_lock so a request arriving mid-print isn't dropped.
            with debug_lock:
                should_print = print_vector_requested
                print_vector_requested = False

            if should_print:
                with state_lock:
                    current_offset = ball_offset
                if current_offset is None:
                    print("[v] No ball currently detected — no vector to print.")
                else:
                    dx, dy = current_offset
                    distance = math.hypot(dx, dy)
                    bearing = math.degrees(math.atan2(dx, -dy))
                    bearing = (bearing + CAMERA_ROTATION_OFFSET) % 360
                    print(
                        f"[v] Ball vector: dx={dx}px, dy={dy}px, "
                        f"distance={distance:.1f}px, bearing={bearing:.1f}deg"
                    )

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
    back into ball_detect_params as new defaults.
    """
    global is_running

    setup_motors()

    def shutdown(sig, frame):
        global is_running
        print("\nShutting down...")
        is_running = False
        stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)

    motor_thread = threading.Thread(target=motor_loop, daemon=True)
    motor_thread.start()

    if ENABLE_WEB_STREAM:
        stream_thread = threading.Thread(target=web_stream_loop, daemon=True)
        stream_thread.start()
        print(f"Web stream at http://<pi-ip>:{STREAM_PORT}")

    print("Ready. Use the AUTO/MANUAL switch on the web page to pick control mode.")
    print("Press q in the video window (or Ctrl+C) to quit.")

    try:
        vision_loop()
    finally:
        is_running = False
        motor_thread.join(timeout=1.0)
        stop()
        with params_lock:
            final_params = dict(ball_detect_params)
        print("\nFinal detection params:")
        for key, value in final_params.items():
            print(f"  {key}: {value}")


if __name__ == "__main__":
    main()