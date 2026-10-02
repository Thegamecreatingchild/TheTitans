import sys
import time
import board
import busio
from armin_config import MotorConfig
from armin_motors import MotorController

cfg = MotorConfig()
addr = int(29, 0)  # accepts 26 or 0x1A
speed = int(sys.argv[2]) if len(sys.argv) > 2 else cfg.max_speed // 10  # default 10%

# Each motor needs ITS OWN calibration, so look it up by address
cals = dict(zip(cfg.addresses, cfg.calibrations))
cals[cfg.dribbler_address] = cfg.dribbler_calibration
if addr not in cals:
    sys.exit(f"No calibration for address {addr}. Known: {sorted(cals)}")

i2c = busio.I2C(board.SCL, board.SDA)
motor = MotorController._create_motor(i2c, addr, cals[addr])

try:
    print(f"Running motor {addr} at speed {speed} for 3s (Ctrl+C to stop)")
    motor.set_speed(speed)
    time.sleep(3)
finally:
    motor.set_speed(0)
    print("Stopped")