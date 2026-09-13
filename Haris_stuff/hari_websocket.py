"""WebSocket control and video broadcasting."""

import asyncio
import json
import time

from hari_config import VisionConfig
from hari_motors import DRIBBLE_KEY, KEY_DEGREES, ROTATE_KEYS, MotorController
from hari_state import RobotState
from hari_vision import VisionService


class WebSocketController:
    def __init__(self, state: RobotState, vision_config: VisionConfig,
                 vision: VisionService, motors: MotorController) -> None:
        self.state = state
        self.vision_config = vision_config
        self.vision = vision
        self.motors = motors

    async def handle(self, websocket) -> None:
        self.state.clients.add(websocket)
        print('Browser connected')
        try:
            async for message in websocket:
                if not isinstance(message, str):
                    continue
                try:
                    data = json.loads(message)
                except json.JSONDecodeError:
                    continue
                self._handle_message(data)
        finally:
            self.state.clients.discard(websocket)
            print('Browser disconnected')

    def _handle_message(self, data: dict) -> None:
        message_type = data.get('type')
        if message_type == 'params':
            self._update_vision_params(data)
        elif message_type == 'mode':
            self._update_mode(data.get('mode'))
        elif message_type == 'keys':
            self._update_keys(data.get('keys', []))
        elif message_type == 'debug_vector':
            self.state.control.print_vector_requested = True
        elif message_type == 'debug_motor':
            enabled = bool(data.get('enabled', not self.state.control.debug_motor))
            self.state.control.debug_motor = enabled
            print(f"[debug] motor debug logging {'ON' if enabled else 'OFF'}")
        elif message_type == 'clahe':
            self._update_clahe(data)

    def _update_vision_params(self, data: dict) -> None:
        tunable = {
            'h_low', 's_low', 'v_low', 'h_high', 's_high', 'v_high',
            'min_contour_area', 'dead_zone_radius', 'ball_dribble_radius',
        }
        for key in tunable:
            if key in data:
                setattr(self.vision_config, key, data[key])

    def _update_mode(self, mode: str) -> None:
        if mode not in ('auto', 'manual'):
            return
        self.state.control.mode = mode
        if mode != 'manual':
            self.state.control.active_keys.clear()

    def _update_keys(self, incoming) -> None:
        valid = {
            key for key in incoming
            if key in KEY_DEGREES or key in ROTATE_KEYS or key == DRIBBLE_KEY
        }
        self.state.control.active_keys = valid
        self.state.control.keys_last_seen = time.time()

    def _update_clahe(self, data: dict) -> None:
        clip_limit = data.get('clip_limit')
        tile_grid = data.get('tile_grid')
        if clip_limit is not None:
            clip_limit = float(clip_limit)
        if tile_grid is not None:
            tile_grid = (int(tile_grid[0]), int(tile_grid[1]))
        self.vision.update_clahe(clip_limit, tile_grid)
        print(
            f"[clahe] clip_limit={self.vision_config.clahe_clip_limit} "
            f"tile_grid={self.vision_config.clahe_tile_grid}"
        )

    async def broadcast(self, payload: bytes) -> None:
        if self.state.clients:
            await asyncio.gather(
                *[client.send(payload) for client in self.state.clients],
                return_exceptions=True,
            )
