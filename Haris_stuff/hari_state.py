"""Live state shared by the robot's asyncio tasks."""

from dataclasses import dataclass, field
from typing import Optional, Set, Tuple


@dataclass
class BallState:
    offset: Optional[Tuple[int, int]] = None
    last_seen: float = 0.0


@dataclass
class ControlState:
    mode: str = 'manual'
    active_keys: Set[str] = field(default_factory=set)
    keys_last_seen: float = 0.0
    print_vector_requested: bool = False
    debug_motor: bool = False


@dataclass
class RobotState:
    ball: BallState = field(default_factory=BallState)
    control: ControlState = field(default_factory=ControlState)
    clients: Set[object] = field(default_factory=set)
    is_running: bool = True
