# WASD.py this is where I control the bot manually.
import math
import time
import threading
import signal
import sys
import curses

import board
import busio
from steelbar_powerful_bldc_driver import PowerfulBLDCDriver

# ----------------------------------------------------------------
# Config
# ----------------------------------------------------------------
MOTOR_ADDRESSES = [25, 26, 27, 28]  # [FL, FR, RL, RR]

SAVED_CAL = [
    {'elecangleoffset': 1435147520, 'sincoscentre': 1241},
    {'elecangleoffset': 1327731200, 'sincoscentre': 1243},
    {'elecangleoffset': 1256835584, 'sincoscentre': 1258},
    {'elecangleoffset': 1150337792, 'sincoscentre': 1247},
]

SPEED = 300000000

# Per-motor speed scale factors, explicitly labeled to match
# MOTOR_ADDRESSES / physical wheel position.
MOTOR_SPEED_SCALE = {
    'FL': 1.0,
    'FR': 1.0,
    'RL': 1.0,
    'RR': 1.0,
}

# Maps each named motor to its index in MOTOR_ADDRESSES / motors[].
MOTOR_INDEX = {
    'FL': 0,
    'FR': 1,
    'RL': 2,
    'RR': 3,
}

# ---- Key -> motor action mapping ----
# Each key contributes a per-motor multiplier in [-1, 1], explicitly
# labeled by motor position (FL/FR/RL/RR) rather than tuple order.
# Contributions from held keys are summed per motor (then clamped to
# [-1, 1]) so combos like w+d blend naturally.
#
#   w / s  -> straight translation, forward / back
#   a / d  -> rotate in place, counter-clockwise / clockwise
#
# NOTE ON THIS HARDWARE: testing showed the left/right-side-split
# pattern (FL,RL vs FR,RR) is what actually drives clean forward/back
# translation on this robot -- not the front/back-axle split you'd
# expect from motor position alone. That axle-split pattern instead
# makes the wheels fight (this is the same 90-degree axis mismatch
# the original script corrected for with TELEOP_ROTATION_OFFSET).
# So translation below uses the side-split pattern, and rotation uses
# a uniform all-motors-same-direction pattern instead, since that's
# the standard in-place-spin pattern for this wheel arrangement.
#
# If forward/back or CW/CCW come out reversed or swapped once tested,
# flip the sign of the relevant dict below -- do not reintroduce a
# global offset, since these two actions are independently defined.
MOTOR_ACTIONS = {
    'w': {'FL': -1, 'FR': 1,  'RL': -1, 'RR': 1},   # forward
    's': {'FL': 1,  'FR': -1, 'RL': 1,  'RR': -1},  # back
    'a': {'FL': 1, 'FR': -1,  'RL': -1, 'RR': 1},   # strafe 
    'd': {'FL': -1,  'FR': 1, 'RL': 1,  'RR': -1},  # rotate CW
    'e': {'FL': -1, 'FR': -1, 'RL': -1, 'RR': -1},  # rotate CCW
    'q': {'FL': 1,  'FR': 1,  'RL': 1,  'RR': 1},   # rotate CW
}

# ================================================================
# TUNE-ME BLOCK
# ================================================================
KEY_LOST_TIMEOUT = 0.15
MOTOR_LOOP_HZ = 50
# ================================================================

state_lock = threading.Lock()
active_keys = set()
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


def move(keys):
    """Drive the motors based on the currently-held key set. Sums each
    held key's per-motor contribution from MOTOR_ACTIONS, clamps each
    motor's total to [-1, 1], then scales by SPEED and that motor's
    MOTOR_SPEED_SCALE entry."""
    totals = {name: 0.0 for name in MOTOR_INDEX}

    for key in keys:
        action = MOTOR_ACTIONS.get(key)
        if action is None:
            continue
        for motor_name, value in action.items():
            totals[motor_name] += value

    for motor_name, index in MOTOR_INDEX.items():
        clamped = max(-1.0, min(1.0, totals[motor_name]))
        speed = int(clamped * SPEED * MOTOR_SPEED_SCALE[motor_name])
        motors[index].set_speed(speed)


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
            keys = set(active_keys)
            seen = last_seen

        now = time.time()

        if not keys or (now - seen) > KEY_LOST_TIMEOUT:
            if moving:
                stop()
                moving = False
        else:
            move(keys)
            moving = True

        time.sleep(period)

    stop()


# ----------------------------------------------------------------
# Keyboard loop (main thread, uses curses for real-time single keys)
# ----------------------------------------------------------------
def keyboard_loop(stdscr):
    global active_keys, last_seen

    curses.cbreak()
    stdscr.keypad(True)
    stdscr.nodelay(True)

    stdscr.addstr(0, 0, "WS = forward/back, AD = rotate left/right, q to quit.")
    stdscr.refresh()

    poll_period = 0.02

    while isRunning:
        pressed_this_tick = set()

        while True:
            ch = stdscr.getch()
            if ch == -1:
                break
            try:
                key = chr(ch).lower()
            except ValueError:
                continue
            if key in MOTOR_ACTIONS:
                pressed_this_tick.add(key)

        if pressed_this_tick:
            with state_lock:
                active_keys = pressed_this_tick
                last_seen = time.time()

        time.sleep(poll_period)


# ----------------------------------------------------------------
# Main
# ----------------------------------------------------------------
def main():
    global isRunning

    setup_motors()

    def shutdown(sig, frame):
        global isRunning
        isRunning = False
        stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)

    motor_thread = threading.Thread(target=motor_loop, daemon=True)
    motor_thread.start()

    print("Starting WASD teleop. Press q (in the terminal window) to quit.")

    try:
        curses.wrapper(keyboard_loop)
    finally:
        isRunning = False
        motor_thread.join(timeout=1.0)
        stop()
        print("\nShutting down...")


if __name__ == "__main__":
    main()