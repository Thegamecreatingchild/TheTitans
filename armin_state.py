"""Live state shared by the robot's asyncio tasks.

These objects contain observations and commands, not configuration. Tasks run
on one asyncio loop, so updates between ``await`` points are kept simple.
"""

from dataclasses import dataclass, field
from typing import Optional, Set, Tuple


@dataclass
class BallState:
    """Most recent ball observation in pixels relative to image centre."""

    offset: Optional[Tuple[int, int]] = None
    last_seen: float = 0.0


@dataclass
class ControlState:
    """Current driving mode, browser keys, and debug requests."""

    mode: str = 'manual'
    active_keys: Set[str] = field(default_factory=set)
    keys_last_seen: float = 0.0
    print_vector_requested: bool = False
    debug_motor: bool = False


@dataclass
class RobotState:
    """All mutable state shared by camera, motor, and WebSocket tasks."""

    ball: BallState = field(default_factory=BallState)
    control: ControlState = field(default_factory=ControlState)
    clients: Set[object] = field(default_factory=set)
    is_running: bool = True
    has_possession : bool = False
