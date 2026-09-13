"""Typed configuration for hardware, vision, networking, and control timing.

Only live tuning values in :class:`VisionConfig` are changed at runtime; the
other configuration objects describe the robot's fixed wiring and startup.
"""

from dataclasses import dataclass, field
from typing import Tuple


@dataclass(frozen=True)
class MotorCalibration:
    """FOC calibration values belonging to one motor driver."""

    elecangleoffset: int
    sincoscentre: int


@dataclass(frozen=True)
class MotorConfig:
    """Motor addresses, calibration, speed limits, and dribbler settings.

    The four drive motors are ordered ``FR, BR, BL, FL`` to match Hari's
    wiring and the index assumptions in ``MotorController.move``.
    """

    addresses: Tuple[int, ...] = (26, 28, 27, 25)
    calibrations: Tuple[MotorCalibration, ...] = (
        MotorCalibration(1327731200, 1241),
        MotorCalibration(1435147520, 1243),
        MotorCalibration(1256835584, 1258),
        MotorCalibration(1150337792, 1247),
    )
    max_speed: int = 80_000_000
    enable_dribbler: bool = False
    dribbler_address: int = 29
    dribbler_calibration: MotorCalibration = MotorCalibration(1437511680, 1245)
    dribbler_speed: int = 80_000_000


@dataclass
class VisionConfig:
    """HSV, geometry, camera-angle, timeout, and CLAHE tuning values."""

    h_low: int = 0
    s_low: int = 200
    v_low: int = 77
    h_high: int = 19
    s_high: int = 255
    v_high: int = 255
    min_contour_area: int = 1
    dead_zone_radius: int = 132
    ball_dribble_radius: int = 145
    camera_rotation_offset: float = 0.0
    ball_lost_timeout: float = 0.3
    clahe_clip_limit: float = 2.5
    clahe_tile_grid: Tuple[int, int] = (8, 8)


@dataclass(frozen=True)
class CameraConfig:
    """Capture format requested from Picamera2."""

    size: Tuple[int, int] = (640, 480)
    fps: int = 120

    @property
    def frame_duration_us(self) -> int:
        return int(1_000_000 / self.fps)


@dataclass(frozen=True)
class NetworkConfig:
    """Network endpoint used by the browser control page."""

    websocket_port: int = 8765


@dataclass(frozen=True)
class ControlConfig:
    """Timing and physical control-input settings."""

    key_lost_timeout: float = 0.3
    motor_loop_delay: float = 0.02
    camera_loop_delay: float = 0.03
    debug_motor: bool = False
    debug_print_hz: int = 5
    movement_switch_gpio: int = 26


@dataclass(frozen=True)
class RobotConfig:
    """Complete startup configuration passed into ``HariApplication``."""

    motors: MotorConfig = field(default_factory=MotorConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    control: ControlConfig = field(default_factory=ControlConfig)
