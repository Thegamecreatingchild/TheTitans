"""Motor setup and movement decisions for Hari's chassis.

This module is the only place that knows the BLDC driver API and Hari's
four-wheel equations. Higher-level code asks it to move, spin, or stop.
"""

import math
import time
from typing import Iterable

import board
import busio

from Old.armin_config import ControlConfig, MotorConfig, VisionConfig
from Old.armin_state import ControlState
from steelbar_powerful_bldc_driver import PowerfulBLDCDriver

class MotorController:
    """Own initialized drive motors and translate decisions into wheel speeds."""

    def __init__(
        self,
        config: MotorConfig,
        vision_config: VisionConfig,
        control_state,
        robot_state,
        control_config: ControlConfig = None,
        debug_hz: int = 5,
    ):
        self.config = config
        self.vision_config = vision_config
        self.control_config = control_config or ControlConfig()
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
        """Create and configure a single motor driver."""
        motor = PowerfulBLDCDriver(i2c, address)
        motor.set_current_limit_foc(65536 * 2) # Set the current limit for the motor
        motor.set_id_pid_constants(1500, 200) # Set the PID constants for current control
        motor.set_speed_pid_constants(4e-2, 4e-4, 3e-2) # Set the PID constants for speed control
        motor.set_ELECANGLEOFFSET(calibration[0]) # Set the ELECANGLEOFFSET register to the calibration value
        motor.set_SINCOSCENTRE(calibration[1]) # Set the SINCOSCENTRE register to the calibration value
        motor.configure_operating_mode_and_sensor(3, 1) # Set to position control mode
        motor.configure_command_mode(12) # Set to speed control mode
        motor.set_speed(0)
        return motor

    def spin_dribbler(self, engaged: bool) -> None:
        """Turn the dribbler on, if it exists"""
        if not self.config.enable_dribbler or self.dribbler_motor is None:
            return
        self.dribbler_motor.set_speed(
            self.config.dribbler_speed if engaged else 0
        )

    def move(self, degree: float, speed: int = None) -> None:
        """Move the bot in a direction, commanding each wheel using math. Check the OneNote to see it."""
        speed = self.config.max_speed if speed is None else speed
        angle_rad = math.radians(degree + 90)
        x = math.floor(math.cos(angle_rad) * speed)
        y = math.floor(math.sin(angle_rad) * speed)
        self.motors[0].set_speed(y + x)
        self.motors[1].set_speed(y - x)
        self.motors[2].set_speed(-(y + x))
        self.motors[3].set_speed(-(y - x))

    def spin(self, speed: int) -> None:
        """Self Explanatory. If you needed to hover over this you really are a dumbass."""
        for motor in self.motors:
            motor.set_speed(speed)

    def stop_wheels(self) -> None:
        """Zero the drive wheels only; the dribbler keeps its current state."""
        for motor in self.motors:
            motor.set_speed(0)

    def stop(self) -> None:
        """Stop every motor including dribbler."""
        for motor in self.motors:
            motor.set_speed(0)
        self.spin_dribbler(False)

    def apply_manual_keys(self, keys: Iterable[str]) -> None:
        """Apply the first matching rotation or translation key from the set."""
        keys = set(keys)
        for key, sign in self.control_config.rotation_keys.items():
            if key in keys:
                speed = sign * int(
                    self.config.max_speed
                    * self.control_config.manual_rotation_speed_ratio
                )
                self._debug(
                    f"[manual] held={sorted(keys)} -> rotate key '{key}' "
                    f"matched first -> spin(speed={speed}) "
                    f"({'CW' if sign > 0 else 'CCW'})"
                )
                self.spin(speed)
                self.spin_dribbler(
                    self.config.enable_dribbler
                    and self.control_config.dribble_key in keys
                )
                return

        if self.control_config.orbit_key in keys:
            arrived = self.orbit_to_behind_ball()
            self._debug(f"[manual] orbit -> arrived={arrived}")
            return
        
        for key, degree in self.control_config.manual_keys.items():
            if key in keys:
                speed = int(
                    self.config.max_speed
                    * self.control_config.manual_translation_speed_ratio
                )
                self._debug(
                    f"[manual] held={sorted(keys)} -> translate key '{key}' "
                    f"matched first -> move(degree={degree}, speed={speed})"
                )
                self.move(degree, speed)
                self.spin_dribbler(
                    self.config.enable_dribbler
                    and self.control_config.dribble_key in keys
                )
                return

        self._debug(f"[manual] held={sorted(keys)} -> no recognised key held -> stop()")
        self.stop()

    def drive_to_the_ball(self, ball_visible: bool, ball_angle: float, distance: float) -> None:
        """Search when the ball is absent; otherwise drive toward its bearing."""
        self._debug("Driving to ball")
        if self.robot_state.has_possession and ball_visible:
            self._debug("[auto] has possession and ball visible -> stop()")
            self.stop_wheels()
            return
        
        if not ball_visible:
            self._debug("[auto] ball not visible (or stale) -> stop()")
            self.stop()
            return

        speed = int(self.config.max_speed * self.config.drive_to_ball_speed_ratio)
        self._debug(
            f"[auto] ball at angle={ball_angle:.1f}deg, distance={distance:.1f}px "
            f"-> move(degree={ball_angle:.1f}, speed={speed})"
        )
        self.move(ball_angle, speed)

    def orbit_to_behind_ball(self, target_bearing: float = 0.0) -> bool:
        """Sweep around the ball with pure translation (no spin) until it lies on
        ``target_bearing`` (robot frame, 0 = ahead; pass the goal bearing to line
        the ball up with the goal). Returns True once arrived."""
        self._debug("Orbiting")
        offset = self.robot_state.ball.offset
        if offset is None:
            self.stop()
            return False

        dx, dy = offset
        distance = math.hypot(dx, dy)
        ball_bearing = (
            math.degrees(math.atan2(dx, -dy)) + self.vision_config.camera_rotation_offset
        ) % 360
        angular_error = ((ball_bearing - target_bearing + 180) % 360) - 180  # signed, (-180, 180]
        
        orbit_radius = self.config.orbit_standoff_radius
        if (
            abs(angular_error) <= self.config.orbit_arrived_angle_tolerance
            and abs(distance - orbit_radius)
            <= orbit_radius * self.config.orbit_arrived_radius_tolerance_ratio
        ):
            self.stop_wheels()
            self._debug("Arrived behind ball")
            return True

        # sweep perpendicular to the ball, easing in as we approach the target angle
        tangent_dir = ball_bearing + (90 if angular_error > 0 else -90)
        ease = min(
            abs(angular_error) / self.config.orbit_full_speed_angle,
            1.0,
        )
        tangent_speed = (
            self.config.max_speed
            * self.config.orbit_max_tangential_speed_ratio
            * ease
        )

        # hold a fixed standoff radius so the sweep doesn't clip or drift from the ball
        radial_error = distance - orbit_radius
        maximum_radial_speed = (
            self.config.max_speed * self.config.orbit_max_radial_speed_ratio
        )
        radial_speed = max(
            -maximum_radial_speed,
            min(
                maximum_radial_speed,
                radial_error * self.config.orbit_radial_gain,
            ),
        )

        tdx, tdy = math.sin(math.radians(tangent_dir)), -math.cos(math.radians(tangent_dir))
        rdx, rdy = math.sin(math.radians(ball_bearing)), -math.cos(math.radians(ball_bearing))
        vx = tangent_speed * tdx + radial_speed * rdx
        vy = tangent_speed * tdy + radial_speed * rdy

        final_bearing = math.degrees(math.atan2(vx, -vy)) % 360
        final_speed = int(min(math.hypot(vx, vy), self.config.max_speed))
        self.move(final_bearing, final_speed)
        return False

    def search_for_goal(self) -> None:
        """Slowly rotate to look for the goal while keeping the ball in the dribbler."""
        self.spin(int(self.config.max_speed * self.config.goal_search_speed_ratio))
        self.spin_dribbler(True)

    def drive_to_goal(self, goal_angle: float, distance: float) -> None:
        """Rotate to face the goal, then drive straight at it with the dribbler on.

        Facing first (instead of translating sideways toward the goal) keeps the
        ball against the dribbler rather than dragging it away.
        """
        error = ((goal_angle + 180) % 360) - 180  # signed; + means goal is clockwise
        if abs(error) > self.config.goal_align_tolerance:
            self._debug(f"[auto] goal at {goal_angle:.1f}deg (err {error:.1f}) -> rotating to face it")
            self.spin(
                int(
                    math.copysign(
                        self.config.max_speed * self.config.goal_rotation_speed_ratio,
                        error,
                    )
                )
            )  # + is CW
        else:
            self._debug(f"[auto] goal aligned, distance={distance:.1f}px -> driving forward")
            self.move(0, int(self.config.max_speed))
        self.spin_dribbler(True)

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