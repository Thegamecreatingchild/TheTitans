"""
Hari's movement code for the robot. It pivots around the ball. Somehow. Magically.

Mk 1
12-09-2026
"""


import asyncio
import cv2
import numpy as np
import math
import json
import websockets
from picamera2 import Picamera2
import sys
import board
import busio
import signal
import time
from steelbar_powerful_bldc_driver import PowerfulBLDCDriver

ball_angle    = 0.0
ball_distance = 0.0
ball_visible  = False

heading_error   = 0.0
goal_heading  = 0.0

POSSESSION_DIST  = 80
POSSESSION_ANGLE = 15
ROTATION_GAIN    = 0.3

ADDRESSES = [26, 28, 27, 25] # FL, FR , BL, BR

SAVED_CAL = [
    {'elecangleoffset': 1327731200, 'sincoscentre': 1241},  # FL
    {'elecangleoffset': 1435147520, 'sincoscentre': 1243},  # FR
    {'elecangleoffset': 1256835584, 'sincoscentre': 1258},  # RL
    {'elecangleoffset': 1150337792, 'sincoscentre': 1247},  # RR
]

MAX_SPEED = 80_000_000
motors = []
i2c = None

def setup_motors():
    global motors, i2c
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
    print("All motors ready")


def setup_camera():
    """
    Gets calibration.json file contents and store certain camera
    settings and the lower and upper orange ball values.
    """
    picam = Picamera2()
    picam.configure(picam.create_preview_configuration())
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

        lower_orange = np.array(calibration["lower_orange"])
        upper_orange = np.array(calibration["upper_orange"])

        print("Loaded calibration.json. Camera locked, using calibrated HSV range.")

    except FileNotFoundError:
        print("No calibration.json found. Run calibrate.py first. Using fallback HSV range for now.")
        lower_orange = np.array([4, 120, 80])
        upper_orange = np.array([24, 255, 255])

    return picam, lower_orange, upper_orange


def move(degree, speed=MAX_SPEED):
    """
    Using trigonometric ratios to work out the correct rotation speed
    for each motor, as well as adding on rotation as a layer of
    movement on the top
    """
    angle_rad = math.radians(degree + 45)
    x = math.floor(math.cos(angle_rad) * speed)
    y = math.floor(math.sin(angle_rad) * speed)
    motors[0].set_speed(y + x)  # FR (25)
    motors[1].set_speed(y - x)  # BR (26)
    motors[2].set_speed(-(y + x))  # BL (27)
    motors[3].set_speed(-(y - x))  # FL (28)


def spin(speed):
    for motor in motors:
        motor.set_speed(speed)


def stop():
    for motor in motors:
        motor.set_speed(0)


clients = set()


async def ws_handler(ws):
    """"
    Handles the connection between camera and browser clients through
    WebSocket
    """
    clients.add(ws)
    print("Browser connected")
    try:
        await ws.wait_closed()
    finally:
        clients.remove(ws)


async def stream_cam(picam, lower, upper):
    global ball_angle, ball_distance, ball_visible

    while True:
        frame = picam.capture_array()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        # Rotated due to actual hardware of camera rotated 90 degrees
        rotated_frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

        height, width = rotated_frame.shape[:2]
        cx, cy = width // 2, height // 2

        hsv = cv2.cvtColor(rotated_frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower, upper)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.circle(rotated_frame, (cx, cy), 4, (0, 255, 0), -1)

        ball_visible = False
        if contours:
            best = max(contours, key=cv2.contourArea)
            if cv2.contourArea(best) > 100:
                mom = cv2.moments(best)
                if mom["m00"] != 0:
                    bx = int(mom["m10"] / mom["m00"])
                    by = int(mom["m01"] / mom["m00"])

                    dx = bx - cx
                    dy = by - cy

                    ball_angle = math.degrees(math.atan2(dx, -dy)) % 360
                    ball_distance = (dx ** 2 + dy ** 2) ** 0.5
                    ball_visible = True

                    cv2.drawContours(rotated_frame, [best], -1, (0, 255, 255), 2)
                    cv2.circle(rotated_frame, (bx, by), 5, (0, 0, 255), -1)
                    cv2.line(rotated_frame, (cx, cy), (bx, by), (255, 0, 0), 2)
                    cv2.putText(rotated_frame,
                                f"angle={round(ball_angle, 1)}  dist={round(ball_distance, 1)}px",
                                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (255, 255, 255), 1)

        success, image_data = cv2.imencode('.jpg', rotated_frame)
        if success and clients:
            jpg = image_data.tobytes()
            await asyncio.gather(*[c.send(jpg) for c in clients])

        await asyncio.sleep(0.03)


async def motor_task():
    while True:
        if not ball_visible:
            spin(int(MAX_SPEED * 0.3))
        else:
            move(ball_angle, int(MAX_SPEED * 0.7))
        await asyncio.sleep(0.02)


async def main():
    setup_motors()
    picam, lower_orange, upper_orange = setup_camera()

    server = await websockets.serve(ws_handler, "0.0.0.0", 8765)
    print("WebSocket stream on port 8765 — open camera.html to watch.")
    print("Ctrl+C to stop.")

    await asyncio.gather(
        stream_cam(picam, lower_orange, upper_orange),
        motor_task(),
    )
    await server.wait_closed()


def shutdown(sig, frame):
    print("\nShutting down...")
    stop()
    sys.exit(0)


if __name__ == "__main__":
    signal.signal(signal.SIGINT, shutdown)
    asyncio.run(main())