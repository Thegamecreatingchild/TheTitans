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
MOTOR_ADDRESSES = [25, 28, 27, 26]  # [FL, FR, RL, RR]

SAVED_CAL = [
    {'elecangleoffset': 1327731200, 'sincoscentre': 1241},
    {'elecangleoffset': 1435147520, 'sincoscentre': 1243},
    {'elecangleoffset': 1256835584, 'sincoscentre': 1258},
    {'elecangleoffset': 1150337792, 'sincoscentre': 1247},
]

# NOTE: this is 12x the 25_000_000 used in the earlier teleop script.
# Double-check it's not a typo before running untethered.
SPEED = 300000000

# ---- Camera mounting compensation ----
# Degrees ADDED to the bearing computed from the image before it is sent
# to move().
#
# With CAMERA_ROTATION_OFFSET = 0 the code assumes:
#   ball toward the TOP of the image  -> robot drives FORWARD (0 deg)
#   ball toward the RIGHT of the image -> robot strafes RIGHT (90 deg)
#
# If the camera is physically rotated 90 deg clockwise (so "image up"
# is actually the robot's right side), set this to 90. If the robot
# drives the wrong way, flip the sign (-90). Any angle works, so a
# camera mounted at 45 deg just means OFFSET = 45.
CAMERA_ROTATION_OFFSET = 0.0

# ================================================================
# TUNE-ME BLOCK - everything you'll adjust during testing lives here
# ================================================================

# px^2: smallest contour accepted as the ball. Raise to reject noise
# specks, lower if the ball is being missed at long range.
MIN_CONTOUR_AREA = 30

# HSV thresholds for the ball (from sphere testing).
LOWER_ORANGE = np.array([7, 139, 141])
UPPER_ORANGE = np.array([10, 255, 255])

# px: ball within this distance of image centre -> stop (prevents jitter)
DEAD_ZONE_RADIUS = 40

# s: stop the motors if the ball hasn't been seen for this long
BALL_LOST_TIMEOUT = 0.3

# motor command update rate (Hz)
MOTOR_LOOP_HZ = 50

# Debug windows (Frame + Mask). Purely for humans - the robot never
# needs them. Set False when running headless / at comp; also saves CPU.
SHOW_WINDOWS = True
# ================================================================

# ----------------------------------------------------------------
# Shared state between vision (main thread) and motors (worker thread)
# ----------------------------------------------------------------
state_lock = threading.Lock()
ball_offset = None   # (dx, dy) in pixels from image centre, or None
last_seen = 0.0      # time.time() of last detection
isRunning = True

motors = []


# ----------------------------------------------------------------
# Motors
# ----------------------------------------------------------------
def setup_motors():
    # General motor setup
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


def move(degree):
    """Translate at full SPEED toward `degree` (0 = forward, 90 = right)."""
    angle_rad = math.radians(degree)

    # Resolves the vector from camera into x and y components
    formula_x = math.floor(math.sin(angle_rad) * SPEED)
    formula_y = math.floor(math.cos(angle_rad) * SPEED)

    # Sets the motors to spin in the correct direction at the right speed
    motors[0].set_speed(formula_y - formula_x)
    motors[1].set_speed(formula_y + formula_x)
    motors[2].set_speed(-(formula_y - formula_x))
    motors[3].set_speed(-(formula_y + formula_x))


def stop():
    for motor in motors:
        motor.set_speed(0)


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
            # Ball lost (or never seen): don't drive blind.
            if moving:
                stop()
                moving = False
        else:
            dx, dy = offset
            dist = math.hypot(dx, dy)

            if dist < DEAD_ZONE_RADIUS:
                # Ball is (roughly) centred under the camera - hold position.
                if moving:
                    stop()
                    moving = False
            else:
                # Image coords: +x right, +y DOWN. atan2(dx, -dy) gives
                # 0 deg when the ball is toward the top of the image and
                # 90 deg to the right - the same convention move() uses.
                bearing = math.degrees(math.atan2(dx, -dy))
                bearing = (bearing + CAMERA_ROTATION_OFFSET) % 360
                move(bearing)
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
            # Converting colors for pi camera
            frame = picamera.capture_array()
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            hsvFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsvFrame, LOWER_ORANGE, UPPER_ORANGE)

            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            # Center of the frame is found as point of reference
            centre_x = frame.shape[1] // 2
            centre_y = frame.shape[0] // 2
            detected = False


            # Finds the biggest orange blob in frame and draws a line to it
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
                            last_seen = time.time() # last moment seen is saved
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
    global isRunning # state of the 

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
        vision_loop()          # blocks on the main thread
    finally:
        isRunning = False
        motor_thread.join(timeout=1.0)
        stop()


if __name__ == "__main__":
    main()
