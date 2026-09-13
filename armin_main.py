"""Entry point and task coordinator for Hari's modular ball-pivot controller.

``HariApplication`` owns the services and connects their async loops; it does
not contain the motor equations, image processing, or message parsing.
"""

import asyncio
import math
import signal
import sys
import time

import cv2
import websockets
from gpiozero import Button

from armin_config import RobotConfig
from armin_motors import MotorController
from armin_state import RobotState
from armin_vision import VisionService
from armin_websocket import WebSocketController


class HariApplication:
    """Compose robot services and coordinate camera, motor, and WebSocket work."""

    def __init__(self, config: RobotConfig = None) -> None:
        self.config = config or RobotConfig()
        self.state = RobotState()
        self.vision = VisionService(self.config.camera, self.config.vision)
        self.motors = MotorController(
            self.config.motors,
            self.config.vision,
            self.state.control,
            self.config.control.debug_print_hz,
        )
        self.websocket = WebSocketController(
            self.state,
            self.config.vision,
            self.vision,
            self.motors,
        )
        self.movement_switch = Button(self.config.control.movement_switch_gpio)
        self.movement_switch.when_pressed = lambda: self.set_mode('auto')
        self.movement_switch.when_released = lambda: self.set_mode('manual')

    def set_mode(self, mode: str) -> None:
        self.state.control.mode = mode
        if mode != 'manual':
            self.state.control.active_keys.clear()
        print(f'[switch] mode set to {mode}')

    async def run(self) -> None:
        """Initialize hardware, serve the browser, and run until shutdown."""
        self.motors.setup()
        picam = self.vision.setup_camera()
        self.set_mode('auto' if self.movement_switch.is_pressed else 'manual')
        server = await websockets.serve(
            self.websocket.handle,
            '0.0.0.0',
            self.config.network.websocket_port,
        )
        print(
            'WebSocket control + video on port '
            f"{self.config.network.websocket_port} — open control.html to drive."
        )
        print('Ctrl+C to stop.')
        try:
            await asyncio.gather(
                self.stream_camera(picam),
                self.motor_loop(),
            )
        finally:
            server.close()
            await server.wait_closed()
            self.motors.stop()

    async def stream_camera(self, picam) -> None:
        """Publish the latest camera observation and JPEG to connected clients."""
        while self.state.is_running:
            frame, offset = self.vision.process_frame(picam)
            if offset is not None:
                self.state.ball.offset = offset
                self.state.ball.last_seen = time.time()
            else:
                self.state.ball.offset = None

            self._print_requested_vector()
            success, encoded = cv2.imencode('.jpg', frame)
            if success:
                await self.websocket.broadcast(encoded.tobytes())
            await asyncio.sleep(self.config.control.camera_loop_delay)

    def _print_requested_vector(self) -> None:
        if not self.state.control.print_vector_requested:
            return
        self.state.control.print_vector_requested = False
        offset = self.state.ball.offset
        if offset is None:
            print('[v] No ball currently detected — no vector to print.')
            return
        dx, dy = offset
        distance = math.hypot(dx, dy)
        bearing = math.degrees(math.atan2(dx, -dy)) % 360
        print(
            f'[v] Ball vector: dx={dx}px, dy={dy}px, '
            f'distance={distance:.1f}px, bearing={bearing:.1f}deg'
        )

    async def motor_loop(self) -> None:
        """Apply manual commands or ball-following decisions at a fixed interval."""
        while self.state.is_running:
            now = time.time()
            control = self.state.control
            if control.mode == 'manual':
                stale = (
                    not control.active_keys
                    or now - control.keys_last_seen > self.config.control.key_lost_timeout
                )
                if stale:
                    self.motors.debug('[manual] no keys held or stale -> stop()')
                    self.motors.stop()
                else:
                    self.motors.apply_manual_keys(control.active_keys)
            else:
                self._apply_auto(now)
            await asyncio.sleep(self.config.control.motor_loop_delay)

    def _apply_auto(self, now: float) -> None:
        """Reject stale observations before handing a valid bearing to the motors."""
        ball = self.state.ball
        timeout = self.config.vision.ball_lost_timeout
        # BallState owns observations; the timeout is configuration, not an observation.
        stale = ball.offset is None or now - ball.last_seen > timeout
        if not stale:
            dx, dy = ball.offset
            distance = math.hypot(dx, dy)
            bearing = math.degrees(math.atan2(dx, -dy)) % 360
            bearing = (bearing + self.config.vision.camera_rotation_offset) % 360
            self.motors.apply_auto(True, bearing, distance)
        else:
            if ball.offset is None:
                self.motors.debug('[auto] no ball detected this frame')
            else:
                self.motors.debug(
                    f'[auto] last detection {now - ball.last_seen:.2f}s ago '
                    f'> BALL_LOST_TIMEOUT ({timeout}s) -> treated as not visible'
                )
            self.motors.apply_auto(False, 0.0, 0.0)

    def shutdown(self) -> None:
        """Stop future loops and remove motor output immediately."""
        self.state.is_running = False
        self.motors.stop()


def main() -> None:
    application = HariApplication()

    def shutdown_handler(_signal, _frame):
        print('\nShutting down...')
        application.shutdown()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown_handler)
    asyncio.run(application.run())


if __name__ == '__main__':
    main()
