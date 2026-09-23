"""
File purpose: TLDR - IMU (BNO085) yaw reader

Returns the robot's yaw in degrees, clockwise = positive, relative to
wherever the bot was pointing at power-on. Call get_yaw() from the motor
loop only (the I2C bus is shared with the motors).
"""

import math

import board
import busio
from adafruit_bno08x import BNO_REPORT_GAME_ROTATION_VECTOR
from adafruit_bno08x.i2c import BNO08X_I2C
from Old.armin_config import ImuConfig


class ImuService:
    def __init__(self, config: ImuConfig = None) -> None:
        self.config = config or ImuConfig()
        self.imu = None

    def setup(self, i2c=None) -> None:
        """Start the IMU. Pass the motors' I2C bus object to share it."""
        if i2c is None:
            i2c = busio.I2C(board.SCL, board.SDA)
        self.imu = BNO08X_I2C(i2c, address=self.config.address)
        self.imu.enable_feature(BNO_REPORT_GAME_ROTATION_VECTOR)
        print("IMU ready")

    def get_yaw(self) -> float:
        """Yaw in degrees, [-180, 180], clockwise positive."""
        i, j, k, real = self.imu.game_quaternion
        yaw = math.degrees(math.atan2(
            2.0 * (real * k + i * j),
            1.0 - 2.0 * (j * j + k * k),
        ))
        return -yaw  # flip to clockwise-positive (assumes Z axis up)