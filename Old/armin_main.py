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

from Old.armin_config import RobotConfig, VisionConfig
from Old.armin_motors import MotorController
from Old.armin_state import RobotState
from Old.armin_vision import VisionService
from Old.armin_websocket import WebSocketController


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
            self.robot_config.control,
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
        """Publish the latest ball + goal observations and JPEG to connected clients."""
        while self.robot_state.is_running:
            frame, ball_offset, goal_offset = self.robot_vision.process_frame(picam)
            now = time.time()

            # Only stamp last_seen on a real detection so staleness checks stay honest.
            self.robot_state.ball.offset = ball_offset
            if ball_offset is not None:
                self.robot_state.ball.last_seen = now
            self.robot_state.goal.offset = goal_offset
            if goal_offset is not None:
                self.robot_state.goal.last_seen = now

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

    @staticmethod
    def _bearing_from_offset(offset, camera_rotation_offset: float) -> float:
        """Convert an image offset into the robot-frame bearing convention."""
        offset_x, offset_y = offset
        return (
            math.degrees(math.atan2(offset_x, -offset_y))
            + camera_rotation_offset
        ) % 360

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
        """Turn fresh ball/goal observations into a single drive decision."""
        state = self.robot_state
        vision = self.robot_config.vision
        motors = self.robot_motors
        ball, goal = state.ball, state.goal
        timeout = vision.ball_lost_timeout

        # Stale if never seen this frame or the last detection is too old.
        if ball.offset is None and now - ball.last_seen > timeout:
            ball_still_visible = None
        else:
            ball_still_visible = ball.offset

        if goal.offset is None and now - goal.last_seen > timeout:
            goal_still_visible = None
        else:
            goal_still_visible = goal.offset

        if ball_still_visible:
            ball_distance = math.hypot(*ball.offset)
            ball_bearing = self._bearing_from_offset(
                ball.offset,
                vision.camera_rotation_offset,
            )
        if goal_still_visible:
            goal_distance = math.hypot(*goal.offset)
            goal_bearing = self._bearing_from_offset(
                goal.offset,
                vision.camera_rotation_offset,
            )

        # Possession is latched: a ball held in the dribbler can sit outside the
        # bot mask and vanish, so it only clears when the ball is seen escaping
        # beyond the capture (orbit) radius.
        if state.has_possession and ball_still_visible and ball_distance > vision.orbit_radius:
            motors.debug('[auto] ball escaped capture radius -> possession cleared')
            state.has_possession = False

        if state.has_possession:
            if goal_still_visible:
                motors.drive_to_goal(goal_bearing, goal_distance)
            else:
                motors.debug('[auto] possession but goal not visible -> search_for_goal()')
                motors.search_for_goal()
            return

        if not ball_still_visible:
            motors.debug(
                '[auto] no ball detected this frame' if ball.offset is None
                else f'[auto] last ball detection {now - ball.last_seen:.2f}s ago '
                     f'> timeout ({timeout}s) -> treated as not visible'
            )
            motors.stop()
            return

        if ball_distance > vision.orbit_radius:
            motors.drive_to_the_ball(True, ball_bearing, ball_distance)
        elif ball_distance > vision.ball_dribble_radius:
            # Line the ball up with the goal so driving at the ball pushes it goalward.
            target_bearing = goal_bearing if goal_still_visible else 0.0
            if motors.orbit_to_behind_ball(target_bearing):
                print('Arrived behind ball, now dribbling')
                state.has_possession = True
                motors.spin_dribbler(True)
        else:
            state.has_possession = True
            motors.spin_dribbler(True)

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