import math
import time
import board
import busio
from adafruit_bno08x import BNO_REPORT_GAME_ROTATION_VECTOR
from adafruit_bno08x.i2c import BNO08X_I2C

i2c = busio.I2C(board.SCL, board.SDA)
imu = BNO08X_I2C(i2c, address=0x4A)
imu.enable_feature(BNO_REPORT_GAME_ROTATION_VECTOR)


def get_yaw_deg():
    x, y, z, w = imu.game_quaternion  # (i, j, k, real)
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return math.degrees(yaw)


while True:
    print(f"yaw: {get_yaw_deg():7.1f} deg")
    time.sleep(0.05)