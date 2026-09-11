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
    'dribbler': 2.0,
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
    # 'dribble': {'dribbler': 1},
    # 'shoot': {'dribbler': -1}
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
    # 'k': {'dribbler': 1},   # dribbler forward
    # 'l': {'dribbler': -1},  # dribbler back
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
    'dead_zone_radius': 140,   # px: contours whose centroid is closer than
                                  # this to image centre are ignored (own body,
                                  # dribbler, reflections). Must be < DEAD_ZONE_RADIUS
                                  # or the bot can never "arrive" at the ball.
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
last_known_position : tuple = None

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
        motor.set_speed_limit(200_000_000)
        motor.set_speed(0)

        motors.append(motor)
        print(f"Motor {index} ready")

    print("All motors ready")

def main():
    setup_motors()
    
    for motor in motors:
        motor.set_speed(0)
    

if __name__ == "__main__":
    main()