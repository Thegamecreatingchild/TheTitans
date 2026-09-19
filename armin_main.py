"""
Author: Aditya Gantimahapatruni
Date created: 18/9/2026

File purpose: TLDR - The file where all other modules are connected to each other.

Description:
This file is the main entry point for the robot application. 
It initializes the robot's configuration, state, vision, motors, and WebSocket controller.
It also sets up a console log tee to forward console output to the WebSocket clients.
"""

import asyncio
import math
import signal
import sys
import time

import cv2
import websockets
from gpiozero import Button

from armin_config import RobotConfig, VisionConfig
from armin_motors import MotorController
from armin_state import RobotState
from armin_vision import VisionService
from armin_websocket import WebSocketController


class ConsoleLogTee:
    """Preserve console output while forwarding complete lines to the UI."""

    def __init__(self, console, on_line) -> None:
        self.console = console
        self.on_line = on_line
        self.pending = ''
    
    def write(self, text: str) -> int:
        """Every printed statement is sent to the console and also forwarded to the WebSocket clients."""
        self.console.write(text)
        self.pending += text
        while '\n' in self.pending:
            line, self.pending = self.pending.split('\n', 1)
            if line:
                self.on_line(line)
        return len(text)

    def flush(self) -> None:
        self.console.flush()


class ArminApplication:
    """Compose robot services and coordinate camera, motor, and WebSocket work."""

    def __init__(self, config: RobotConfig = None) -> None:
        self.robot_config = config or RobotConfig()
        self.robot_state = RobotState()
        self.robot_vision = VisionService(self.robot_config.camera, self.robot_config.vision)
        self.robot_motors = MotorController(
            self.robot_config.motors,
            self.robot_config.vision,
            self.robot_state.control,
            self.robot_state,
            self.robot_config.control.debug_print_hz,
        )
        self.robot_websocket = WebSocketController(
            self.robot_state,
            self.robot_config.vision,
            self.robot_vision,
            self.robot_motors,
        )
        self.original_stdout = sys.stdout
        sys.stdout = ConsoleLogTee(sys.stdout, self.robot_websocket.record_log) # records smth for websockets
        self.movement_switch = Button(self.robot_config.control.movement_switch_gpio)
        self.movement_switch.when_pressed = lambda: self.set_mode('auto')
        self.movement_switch.when_released = lambda: self.set_mode('manual')

    def set_mode(self, mode: str) -> None:
        self.robot_state.control.mode = mode
        if mode != 'manual':
            self.robot_state.control.active_keys.clear()
        print(f'[switch] mode set to {mode}')

    async def run(self) -> None:
        """Initialize hardware, serve the browser, and run until shutdown."""
        loop = asyncio.get_running_loop()
        loop.add_signal_handler(signal.SIGINT, self._on_sigint)
        self.robot_websocket.set_loop(asyncio.get_running_loop())
        self.robot_motors.setup()
        picam = self.robot_vision.setup_camera()
        self.set_mode('auto' if self.movement_switch.is_pressed else 'manual')
        server = await websockets.serve(
            self.robot_websocket.handle,
            '0.0.0.0',
            self.robot_config.network.websocket_port,
        )
        print(
            'WebSocket control + video on port '
            f"{self.robot_config.network.websocket_port} — open control.html to drive."
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
            self.robot_motors.stop()
            sys.stdout = self.original_stdout

    async def stream_camera(self, picam) -> None:
        """Publish the latest camera observation and JPEG to connected clients."""
        last_target = None
        while self.robot_state.is_running:
            target = 'yellow_goal' if self.robot_state.has_possession else 'ball' # target is set, ball or goal depending on possession
            if target != last_target:
                print(target)
                last_target = target
            
            frame, offset = self.robot_vision.process_frame(picam, target) # searches for a target
            if target == 'ball': # If it can see the ball
                self.robot_state.ball.offset = offset
                self.robot_state.ball.last_seen = time.time()
            else:
                self.robot_state.goal.offset = None
                self.robot_state.goal.last_seen = time.time()

            self._print_requested_vector()
            success, encoded = cv2.imencode('.jpg', frame)
            if success:
                await self.robot_websocket.broadcast(encoded.tobytes())
            await asyncio.sleep(self.robot_config.control.camera_loop_delay)
    
    def _print_requested_vector(self) -> None:
        if not self.robot_state.control.print_vector_requested:
            return
        self.robot_state.control.print_vector_requested = False
        offset = self.robot_state.ball.offset
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
        while self.robot_state.is_running:
            now = time.time()
            control = self.robot_state.control
            if control.mode == 'manual':
                stale = (
                    not control.active_keys
                    or now - control.keys_last_seen > self.robot_config.control.key_lost_timeout
                )
                if stale:
                    self.robot_motors.debug('[manual] no keys held or stale -> stop()')
                    self.robot_motors.stop()
                else:
                    self.robot_motors.apply_manual_keys(control.active_keys)
            else:
                self._apply_auto(now)
            await asyncio.sleep(self.robot_config.control.motor_loop_delay)

    def _apply_auto(self, now: float) -> None:
        """Reject stale observations before handing a valid bearing to the motors."""
        ball = self.robot_state.ball
        timeout = self.robot_config.vision.ball_lost_timeout
        # BallState owns observations; the timeout is configuration, not an observation.
        # stale if last ball detect call is too old
        stale = ball.offset is None or now - ball.last_seen > timeout
        if not stale:
            dx, dy = ball.offset
            distance = math.hypot(dx, dy)
            bearing = math.degrees(math.atan2(dx, -dy)) % 360
            bearing = (bearing + self.robot_config.vision.camera_rotation_offset) % 360 # Update for camera offset
            
            # Possession check - if the ball is outside deadzone and inside dribble radius.
            if distance <= self.robot_config.vision.orbit_radius:
                print("Start orbiting")
                if distance > self.robot_config.vision.ball_dribble_radius:
                    arrived = self.robot_motors.orbit_to_behind_ball()
                    if arrived and distance < self.robot_config.vision.ball_dribble_radius:
                        self.robot_state.has_possession = True
                        self.robot_motors.spin_dribbler(True)
                            
            elif self.robot_state.has_possession:
                if distance > self.robot_config.vision.goal_stop_distance:
                    self.robot_motors.drive_to_goal(True, bearing, distance)
                else:
                    self.robot_motors.debug(f"[auto] has possession and ball is close enough -> stop()")
                    self.robot_motors.stop()
                
            else:
                print("Get closer")
                self.robot_state.has_possession = False
                self.robot_motors.drive_to_the_ball(True, bearing, distance)
        else:
            if ball.offset is None:
                self.robot_motors.debug('[auto] no ball detected this frame')
            else:
                self.robot_motors.debug(
                    f'[auto] last detection {now - ball.last_seen:.2f}s ago '
                    f'> BALL_LOST_TIMEOUT ({timeout}s) -> treated as not visible'
                )
            self.robot_motors.stop()

    def _on_sigint(self) -> None:
        print('\nShutting down...')
        self.shutdown()   # sets is_running=False and stops motors

    def shutdown(self) -> None:
        """Stop future loops and remove motor output immediately."""
        self.robot_state.is_running = False
        self.robot_motors.stop()


def main() -> None:
    application = ArminApplication()
    try:
        asyncio.run(application.run())
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
