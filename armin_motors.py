"""Motor setup and movement decisions for Hari's chassis.

This module is the only place that knows the BLDC driver API and Hari's
four-wheel equations. Higher-level code asks it to move, spin, or stop.
"""

import math
import time
from typing import Iterable, Set

import board
import busio

from armin_config import MotorConfig, VisionConfig
from armin_state import ControlState
from steelbar_powerful_bldc_driver import PowerfulBLDCDriver

# Translation keys use the same bearing convention as the camera: 0 is ahead.
KEY_DEGREES = {'w': 0, 'd': 90, 's': 180, 'a': 270}
# Rotate keys use the same convention as the camera: 1 is clockwise, -1 is counterclockwise.
ROTATE_KEYS = {'q': 1, 'e': -1}
DRIBBLE_KEY = 'k'


class MotorController:
    """Own initialized drive motors and translate decisions into wheel speeds."""

    def __init__(self, config, vision_config, control_state, robot_state, debug_hz=5):
        self.config = config
        self.vision_config = vision_config
        self.control_state = control_state
        self.robot_state = robot_state
        self.motors = []
        self.dribbler_motor = None
        self._debug_last_print = 0.0
        self.debug_hz = debug_hz

    def setup(self) -> None:
        """Create and configure the four drive drivers and optional dribbler."""
        i2c = busio.I2C(board.SCL, board.SDA)
        for index, address in enumerate(self.config.addresses):
            motor = self._create_motor(i2c, address, self.config.calibrations[index])
            self.motors.append(motor)
            print(f"Motor {index} ready")

        if self.config.enable_dribbler:
            self.dribbler_motor = self._create_motor(
                i2c,
                self.config.dribbler_address,
                self.config.dribbler_calibration,
            )
            print("Dribbler motor ready")

        print("All motors ready")

    @staticmethod
    def _create_motor(i2c, address, calibration):
        motor = PowerfulBLDCDriver(i2c, address)
        motor.set_current_limit_foc(65536 * 2)
        motor.set_id_pid_constants(1500, 200)
        motor.set_iq_pid_constants(1500, 200)
        motor.set_speed_pid_constants(4e-2, 4e-4, 3e-2)
        motor.set_ELECANGLEOFFSET(calibration.elecangleoffset)
        motor.set_SINCOSCENTRE(calibration.sincoscentre)
        motor.configure_operating_mode_and_sensor(3, 1)
        motor.configure_command_mode(12)
        motor.set_speed(0)
        return motor

    def set_dribbler(self, engaged: bool) -> None:
        if not self.config.enable_dribbler or self.dribbler_motor is None:
            return
        self.dribbler_motor.set_speed(
            -self.config.dribbler_speed if engaged else 0
        )

    def move(self, degree: float, speed: int = None) -> None:
        """Drive in a compass direction using Hari's math."""
        speed = self.config.max_speed if speed is None else speed
        angle_rad = math.radians(degree + 90)
        x = math.floor(math.cos(angle_rad) * speed)
        y = math.floor(math.sin(angle_rad) * speed)
        self.motors[0].set_speed(y + x)
        self.motors[1].set_speed(y - x)
        self.motors[2].set_speed(-(y + x))
        self.motors[3].set_speed(-(y - x))

    def spin(self, speed: int) -> None:
        """Turn in place by commanding every drive wheel equally."""
        for motor in self.motors:
            motor.set_speed(speed)

    def stop(self) -> None:
        """Stop every motor and release the dribbler."""
        for motor in self.motors:
            motor.set_speed(0)
        self.set_dribbler(False)

    def apply_manual_keys(self, keys: Iterable[str]) -> None:
        """Apply the first matching rotation or translation key from the set."""
        keys = set(keys)
        for key, sign in ROTATE_KEYS.items():
            if key in keys:
                speed = sign * int(self.config.max_speed * 0.3)
                self._debug(
                    f"[manual] held={sorted(keys)} -> rotate key '{key}' "
                    f"matched first -> spin(speed={speed}) "
                    f"({'CW' if sign > 0 else 'CCW'})"
                )
                self.spin(speed)
                self.set_dribbler(self.config.enable_dribbler and DRIBBLE_KEY in keys)
                return

        for key, degree in KEY_DEGREES.items():
            if key in keys:
                speed = int(self.config.max_speed * 0.7)
                self._debug(
                    f"[manual] held={sorted(keys)} -> translate key '{key}' "
                    f"matched first -> move(degree={degree}, speed={speed})"
                )
                self.move(degree, speed)
                self.set_dribbler(self.config.enable_dribbler and DRIBBLE_KEY in keys)
                return

        self._debug(f"[manual] held={sorted(keys)} -> no recognised key held -> stop()")
        self.stop()

    def apply_auto(self, ball_visible: bool, ball_angle: float, distance: float) -> None:
        """Search when the ball is absent; otherwise drive toward its bearing."""
        if self.robot_state.has_possession and ball_visible:
            self._debug(f"[auto] has possession and ball visible -> stop()")
            self.spin(self.config.max_speed // 2)
            return
        
        if not ball_visible:
            speed = int(self.config.max_speed * 0.3)
            self._debug(f"[auto] ball not visible (or stale) -> spin(speed={speed}) to search")
            self.spin(speed)
            self.set_dribbler(False)
            return

        speed = int(self.config.max_speed * 0.7)
        self._debug(
            f"[auto] ball at angle={ball_angle:.1f}deg, distance={distance:.1f}px "
            f"-> move(degree={ball_angle:.1f}, speed={speed})"
        )
        self.move(ball_angle, speed)
        if distance <= self.vision_config.ball_dribble_radius:
            self._debug(
                f"[auto] distance {distance:.1f}px <= dribble radius "
                f"{self.vision_config.ball_dribble_radius}px -> dribbler ENGAGE"
                f"{'' if self.config.enable_dribbler else ' (skipped, enable_dribbler=False)'}"
            )
            self.set_dribbler(True)
        else:
            self.set_dribbler(False)

    def debug(self, message: str) -> None:
        self._debug(message)

    def _debug(self, message: str) -> None:
        if not self.control_state.debug_motor:
            return
        now = time.time()
        if now - self._debug_last_print < 1.0 / self.debug_hz:
            return
        self._debug_last_print = now
        print(message)
