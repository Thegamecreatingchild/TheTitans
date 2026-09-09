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

# ----------------------------------------------------------------
# Config
# ----------------------------------------------------------------
MOTOR_ADDRESSES = [25, 26, 27, 28]  # [FL, FR, RL, RR]

SAVED_CAL = [
    {'elecangleoffset': 1327731200, 'sincoscentre': 1241},
    {'elecangleoffset': 1435147520, 'sincoscentre': 1243},
    {'elecangleoffset': 1256835584, 'sincoscentre': 1258},
    {'elecangleoffset': 1150337792, 'sincoscentre': 1247},
]

SPEED = 300000000

# Explicitly labeled per-motor scale, matching the WASD script.
MOTOR_SPEED_SCALE = {
    'FL': 1.0,
    'FR': 0.50,
    'RL': 0.50,
    'RR': 0.50,
}

MOTOR_INDEX = {
    'FL': 0,
    'FR': 1,
    'RL': 2,
    'RR': 3,
}

# ---- Motor action patterns ----
# Same patterns validated in the WASD teleop script for this chassis:
# side-split (FL/RL vs FR/RR) drives clean forward/back translation,
# and uniform all-motors-same-direction drives in-place rotation.
MOTOR_ACTIONS = {
    'forward': {'FL': -1, 'FR': 1,  'RL': -1, 'RR': 1},
    'back':    {'FL': 1,  'FR': -1, 'RL': 1,  'RR': -1},
    'ccw':     {'FL': -1, 'FR': -1, 'RL': -1, 'RR': -1},
    'cw':      {'FL': 1,  'FR': 1,  'RL': 1,  'RR': 1},
}

# ---- Camera mounting compensation ----
# Degrees ADDED to the bearing computed from the image before it is
# used for heading control.
#
# With CAMERA_ROTATION_OFFSET = 0 the code assumes:
#   ball toward the TOP of the image   -> bearing 0 (robot's forward)
#   ball toward the RIGHT of the image -> bearing 90 (robot's right)
CAMERA_ROTATION_OFFSET = 0.0

# ================================================================
# TUNE-ME BLOCK - everything you'll adjust during testing lives here
# ================================================================

# px^2: smallest contour accepted as the ball.
MIN_CONTOUR_AREA = 6

# HSV thresholds for the ball (from sphere testing).
LOWER_ORANGE = np.array([7, 139, 141])
UPPER_ORANGE = np.array([10, 255, 255])

# px: ball within this distance of image centre -> stop (prevents jitter)
DEAD_ZONE_RADIUS = 140

# deg: how far off "straight ahead" the ball can be before we drive
# forward without rotating first (0 = must be perfectly aligned).
FORWARD_ALIGN_TOLERANCE = 15.0

# deg: how close to directly-behind (180) the ball has to be before
# we just reverse instead of rotating 180 to face it first.
BACKWARD_ALIGN_TOLERANCE = 15.0

# s: stop the motors if the ball hasn't been seen for this long
BALL_LOST_TIMEOUT = 0.3

# motor command update rate (Hz)
MOTOR_LOOP_HZ = 50

SHOW_WINDOWS = False
# ================================================================

state_lock = threading.Lock()
ball_offset = None
last_seen = 0.0
isRunning = True

motors = []


# ----------------------------------------------------------------
# Motors
# ----------------------------------------------------------------
def setup_motors():
    global motors
    i2c = busio.I2C(board.SCL, board.SDA)

    for index, address in enumerate(MOTOR_ADDRESSES):
        motor = PowerfulBLDCDriver(i2c, address)

        motor.set_current_limit_foc(65536)
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

    print("All motors ready")


def apply_action(action_name):
    """Drive the motors at full SPEED for a single named action
    ('forward', 'back', 'ccw', 'cw'), using MOTOR_ACTIONS."""
    pattern = MOTOR_ACTIONS[action_name]
    for motor_name, index in MOTOR_INDEX.items():
        speed = int(pattern[motor_name] * SPEED * MOTOR_SPEED_SCALE[motor_name])
        motors[index].set_speed(speed)


def stop():
    for motor in motors:
        motor.set_speed(0)


def signed_heading_error(bearing):
    """Convert a 0-360 bearing (0 = forward) into a signed error in
    [-180, 180]: positive = ball is to the right, negative = left."""
    return ((bearing + 180) % 360) - 180


def choose_action(bearing):
    """Decide a single motor action from the ball's bearing.
    - Roughly ahead -> drive forward.
    - Roughly directly behind -> just reverse (no point turning 180).
    - Otherwise -> rotate (shortest way) to face the ball first."""
    error = signed_heading_error(bearing)

    if abs(error) <= FORWARD_ALIGN_TOLERANCE:
        return 'forward'

    if abs(abs(error) - 180) <= BACKWARD_ALIGN_TOLERANCE:
        return 'back'

    return 'cw' if error > 0 else 'ccw'


# ----------------------------------------------------------------
# Motor control loop (worker thread)
# ----------------------------------------------------------------
def motor_loop():
    period = 1.0 / MOTOR_LOOP_HZ
    moving = False

    while isRunning:
        with state_lock:
            offset = ball_offset
            seen = last_seen

        now = time.time()

        if offset is None or (now - seen) > BALL_LOST_TIMEOUT:
            if moving:
                stop()
                moving = False
        else:
            dx, dy = offset
            dist = math.hypot(dx, dy)

            if dist < DEAD_ZONE_RADIUS:
                if moving:
                    stop()
                    moving = False
            else:
                bearing = math.degrees(math.atan2(dx, -dy))
                bearing = (bearing + CAMERA_ROTATION_OFFSET) % 360
                apply_action(choose_action(bearing))
                moving = True

        time.sleep(period)

    stop()


# ----------------------------------------------------------------
# Vision loop (main thread, so cv2.imshow behaves)
# ----------------------------------------------------------------
def vision_loop():
    global ball_offset, last_seen

    picamera = Picamera2()
    picamera.configure(picamera.create_preview_configuration())
    picamera.start()

    try:
        while isRunning:
            frame = picamera.capture_array()
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            hsvFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsvFrame, LOWER_ORANGE, UPPER_ORANGE)

            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            centre_x = frame.shape[1] // 2
            centre_y = frame.shape[0] // 2
            detected = False

            if contours:
                biggest_contour = max(contours, key=cv2.contourArea)
                contour_area = cv2.contourArea(biggest_contour)

                if contour_area > MIN_CONTOUR_AREA:
                    moments = cv2.moments(biggest_contour)
                    if moments['m00'] != 0:
                        cX = int(moments['m10'] / moments['m00'])
                        cY = int(moments['m01'] / moments['m00'])

                        with state_lock:
                            ball_offset = (cX - centre_x, cY - centre_y)
                            last_seen = time.time()
                        detected = True

                        if SHOW_WINDOWS:
                            cv2.circle(frame, (cX, cY), 5, (0, 255, 0), -1)
                            cv2.line(frame, (centre_x, centre_y),
                                     (cX, cY), (0, 0, 255), 2)

            if not detected:
                with state_lock:
                    ball_offset = None

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

    print("Following ball. Press q in the video window (or Ctrl+C) to quit.")

    try:
        vision_loop()
    finally:
        isRunning = False
        motor_thread.join(timeout=1.0)
        stop()


if __name__ == "__main__":
    main()