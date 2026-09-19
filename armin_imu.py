"""
Author: Aditya Gantimahapatruni
Date created: 19/9/2026

File purpose: TLDR - IMU (BNO085) yaw reader for heading hold

Description:
Owns the BNO085 and exposes one number: the robot's yaw in degrees, with
clockwise = positive (the same convention as the camera bearing and
ROTATE_KEYS in armin_motors.py).

Uses the game rotation vector (gyro + accelerometer fused, NO magnetometer),
so the BLDC motors can't disturb it. Yaw is relative to wherever the bot was
when it powered on, so "forward" is captured at runtime, not assumed.

Thread safety: call get_yaw() from ONE place only (the motor loop). The I2C
bus is shared with the motor drivers and Blinka is not thread-safe, so never
call this from the gpiozero button callback or a WebSocket handler.
"""

import math
import time
from typing import Optional

import board
import busio
from adafruit_bno08x import BNO_REPORT_GAME_ROTATION_VECTOR
from adafruit_bno08x.i2c import BNO08X_I2C


def wrap_degrees(angle: float) -> float:
    """Wrap any angle to the range [-180, 180)."""
    return (angle + 180.0) % 360.0 - 180.0


def heading_error(target: float, current: float) -> float:
    """Signed shortest error from current to target, in degrees.

    Positive means the bot must turn clockwise to reach the target.
    """
    return wrap_degrees(target - current)


class ImuService:
    """Read yaw from a BNO085, riding out the occasional bad I2C read."""

    def __init__(self, address: int = 0x4A, clockwise_positive: bool = True,
                 startup_timeout: float = 3.0) -> None:
        """
        address:            I2C address (i2cdetect showed 0x4A).
        clockwise_positive: True gives clockwise-positive yaw, assuming the
                            chip's Z axis points UP. If turning the bot
                            clockwise by hand makes the yaw go DOWN, flip
                            this. If yaw barely changes when you rotate, the
                            board is mounted on its side/upside down and the
                            axes need changing.
        startup_timeout:    seconds to wait for the first valid reading.
        """
        self.address = address
        self.clockwise_positive = clockwise_positive
        self.startup_timeout = startup_timeout
        self._imu: Optional[BNO08X_I2C] = None
        self._last_yaw: Optional[float] = None
        self._fail_count = 0

    @property
    def healthy(self) -> bool:
        """False after several consecutive failed reads (wiring/bus trouble)."""
        return self._imu is not None and self._fail_count < 5

    def setup(self, i2c=None) -> None:
        """Start the IMU and block until the first reading arrives.

        Pass the SAME busio.I2C object the motors use (share one bus object
        rather than creating a second). If omitted, a new one is created.
        Hold the bot still while this runs.
        """
        if i2c is None:
            i2c = busio.I2C(board.SCL, board.SDA)
        self._imu = BNO08X_I2C(i2c, address=self.address)
        self._imu.enable_feature(BNO_REPORT_GAME_ROTATION_VECTOR)

        deadline = time.time() + self.startup_timeout
        while time.time() < deadline:
            if self._read_yaw_raw() is not None:
                print("IMU ready")
                return
            time.sleep(0.05)
        raise RuntimeError(
            f"IMU at 0x{self.address:02x} gave no reading within "
            f"{self.startup_timeout}s — check wiring / i2cdetect."
        )

    def _read_yaw_raw(self) -> Optional[float]:
        """One read attempt. Returns yaw in degrees, or None on failure."""
        try:
            i, j, k, real = self._imu.game_quaternion
            yaw = math.degrees(math.atan2(
                2.0 * (real * k + i * j),
                1.0 - 2.0 * (j * j + k * k),
            ))
        except Exception:
            # I2C hiccup, no packet yet, malformed report, etc.
            self._fail_count += 1
            return None
        self._fail_count = 0
        if self.clockwise_positive:
            yaw = -yaw
        self._last_yaw = yaw
        return yaw

    def get_yaw(self) -> Optional[float]:
        """Latest yaw in degrees ([-180, 180], clockwise positive).

        On a failed read this returns the last good value so a single bad
        transaction can't crash the motor loop. Returns None only if there
        has never been a good reading. Check `healthy` if you want to stop
        trusting a stale value.
        """
        yaw = self._read_yaw_raw()
        return yaw if yaw is not None else self._last_yaw


if __name__ == "__main__":
    # Standalone check: rotate the bot by hand and watch the error.
    # Clockwise should make yaw go UP and the error go DOWN (error is
    # target - current, relative to the heading captured at startup).
    imu = ImuService()
    imu.setup()
    forward = imu.get_yaw()
    print(f"Captured forward = {forward:.1f} deg")
    while True:
        yaw = imu.get_yaw()
        print(
            f"yaw={yaw:7.1f}  error to forward={heading_error(forward, yaw):7.1f}"
            f"  healthy={imu.healthy}"
        )
        time.sleep(0.1)