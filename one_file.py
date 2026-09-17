import asyncio
import json
import math
import time
import signal
import sys
from gpiozero import Button

import cv2
import numpy as np
import board
import busio
import websockets
from picamera2 import Picamera2
from steelbar_powerful_bldc_driver import PowerfulBLDCDriver

# ---------------- CONFIG ----------------
MOTOR_ADDRESSES = [26, 28, 27, 25]          # order: FR, BR, BL, FL
SAVED_CAL = [
    {"elecangleoffset": 1327731200, "sincoscentre": 1241},
    {"elecangleoffset": 1435147520, "sincoscentre": 1243},
    {"elecangleoffset": 1256835584, "sincoscentre": 1258},
    {"elecangleoffset": 1150337792, "sincoscentre": 1247},
]
MAX_SPEED = 40_000_000

LOWER_ORANGE = np.array([0, 200, 77])
UPPER_ORANGE = np.array([19, 255, 255])
MIN_CONTOUR_AREA = 1
DEAD_ZONE_RADIUS = 128
BALL_LOST_TIMEOUT = 0.3

WS_PORT = 8765
KEY_DEGREES = {"w": 0, "d": 90, "s": 180, "a": 270}
ROTATE_KEYS = {"q": -1, "e": 1}

# ---------------- SHARED STATE ----------------
motors = []
ball_offset = None
last_seen = 0.0
mode = "auto"          # "auto" or "manual"
active_keys = set()
clients = set()

switch = Button(26)  # GPIO pin 26 for the switch

def toggle_control_mode():
    global mode
    if mode == "auto":
        mode = "manual"
        print("Switched to manual control mode.")
    else:
        mode = "auto"
        print("Switched to automatic control mode.")

switch.when_pressed = toggle_control_mode

# ---------------- MOTORS ----------------
def setup_motors():
    global motors
    i2c = busio.I2C(board.SCL, board.SDA)
    for index, address in enumerate(MOTOR_ADDRESSES):
        motor = PowerfulBLDCDriver(i2c, address)
        motor.set_current_limit_foc(131072)
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


def move(degree, speed=MAX_SPEED):
    angle_rad = math.radians(degree + 90)
    x = math.floor(math.cos(angle_rad) * speed)
    y = math.floor(math.sin(angle_rad) * speed)
    motors[0].set_speed(y + x)
    motors[1].set_speed(y - x)
    motors[2].set_speed(-(y + x))
    motors[3].set_speed(-(y - x))


def spin(speed):
    for motor in motors:
        motor.set_speed(speed)


def stop():
    for motor in motors:
        motor.set_speed(0)


# ---------------- VISION ----------------
def detect_ball(frame):
    frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    height, width = frame.shape[:2]
    cx, cy = width // 2, height // 2

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, LOWER_ORANGE, UPPER_ORANGE)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    offset = None
    if contours:
        best = max(contours, key=cv2.contourArea)
        if cv2.contourArea(best) > MIN_CONTOUR_AREA:
            m = cv2.moments(best)
            if m["m00"] != 0:
                bx = int(m["m10"] / m["m00"])
                by = int(m["m01"] / m["m00"])
                
                if math.hypot(bx - cx, by - cy) < DEAD_ZONE_RADIUS:
                    return frame, None  # Ball is in the dead zone; ignore it
                
                if math.hypot(bx - cx, by - cy) >= DEAD_ZONE_RADIUS:
                    offset = (bx - cx, by - cy)
                cv2.circle(frame, (bx, by), 5, (0, 0, 255), -1)
                cv2.line(frame, (cx, cy), (bx, by), (255, 0, 0), 2)

    cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)
    return frame, offset


def setup_camera():
    picam = Picamera2()
    picam.configure(picam.create_preview_configuration(main={"size": (640, 480)}))
    picam.start()
    try:
        with open("calibration.json") as f:
            cal = json.load(f)
        picam.set_controls({
            "AeEnable": False,
            "AwbEnable": False,
            "ExposureTime": cal["exposure_time"],
            "AnalogueGain": cal["analogue_gain"],
            "ColourGains": tuple(cal["colour_gains"]),
        })
        print("Loaded calibration.json.")
    except FileNotFoundError:
        print("No calibration.json found — using default camera settings.")
    return picam


# ---------------- MAIN LOOPS ----------------
async def vision_loop(picam):
    global ball_offset, last_seen
    while True:
        frame = picam.capture_array()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        frame, offset = detect_ball(frame)

        if offset is not None:
            ball_offset = offset
            last_seen = time.time()
        else:
            ball_offset = None

        success, jpg = cv2.imencode(".jpg", frame)
        if success and clients:
            await asyncio.gather(*[c.send(jpg.tobytes()) for c in clients])

        await asyncio.sleep(0.03)


async def motor_loop():
    while True:
        if mode == "manual":
            apply_manual_keys()
        else:
            apply_auto()
        await asyncio.sleep(0.02)


def apply_manual_keys():
    for key, sign in ROTATE_KEYS.items():
        if key in active_keys:
            spin(sign * int(MAX_SPEED * 0.3))
            return
    for key, degree in KEY_DEGREES.items():
        if key in active_keys:
            move(degree, int(MAX_SPEED * 0.7))
            return
    stop()


def apply_auto():
    if ball_offset is None or (time.time() - last_seen) > BALL_LOST_TIMEOUT:
        spin(int(MAX_SPEED * 0.3))   # ball not visible -> search by spinning
        return
    dx, dy = ball_offset
    bearing = math.degrees(math.atan2(dx, -dy)) % 360
    move(bearing, int(MAX_SPEED * 0.7))


# ---------------- WEBSOCKET (browser control + video) ----------------
async def ws_handler(ws):
    global mode
    clients.add(ws)
    print("Browser connected")
    try:
        async for message in ws:
            try:
                data = json.loads(message)
            except json.JSONDecodeError:
                continue
            if data.get("type") == "mode":
                mode = data.get("mode", mode)
            elif data.get("type") == "keys":
                active_keys.clear()
                active_keys.update(data.get("keys", []))
    finally:
        clients.discard(ws)
        print("Browser disconnected")


# ---------------- ENTRY POINT ----------------
async def main():
    setup_motors()
    picam = setup_camera()

    server = await websockets.serve(ws_handler, "0.0.0.0", WS_PORT)
    print(f"WebSocket on port {WS_PORT} — open control.html to drive.")
    print("Ctrl+C to stop.")

    await asyncio.gather(
        vision_loop(picam),
        motor_loop(),
    )


def shutdown(sig, frame):
    print("\nShutting down...")
    stop()
    sys.exit(0)


if __name__ == "__main__":
    signal.signal(signal.SIGINT, shutdown)
    asyncio.run(main())