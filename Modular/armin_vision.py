"""
Author: Aditya Gantimahapatruni
Date created: 18/9/2026

File purpose: TLDR - A Camera Handler

Description:
This file contains a class that handles most vision logic, including boosting saturation of orange pixels,
finding a ball, and returning its position to allow other modules to operate with new info.

"""

from pathlib import Path
import math
from typing import Optional, Tuple

import cv2
import numpy as np
from picamera2 import Picamera2

from armin_config import CameraConfig, VisionConfig


class VisionService:
    """Configure Picamera2 and turn frames into annotated images plus ball offsets."""

    def __init__(self, camera_config: CameraConfig, config: VisionConfig) -> None:
        self.camera_config = camera_config
        self.config = config
        self.valid_mask = cv2.imread(config.valid_mask_path, cv2.IMREAD_GRAYSCALE)
        if self.valid_mask is None:
            raise FileNotFoundError(
            f"{config.valid_mask_path} not found or unreadable — "
                "generate it before starting VisionService."
            )
        self.valid_mask = cv2.rotate(self.valid_mask, cv2.ROTATE_90_COUNTERCLOCKWISE)
        self._clahe = self._create_clahe()

    def setup_camera(self) -> Picamera2:
        """Start the camera and apply configured exposure/white-balance settings."""
        picam = Picamera2()
        config = picam.create_video_configuration(
            main={'size': self.camera_config.size},
            controls={
                'FrameDurationLimits': (
                    self.camera_config.frame_duration_us,
                    self.camera_config.frame_duration_us,
                )
            },
        )
        picam.configure(config)
        picam.start()
        # Setting the camera's fps to be higher by reducing the exposure time and various others.
        print("valid_mask:", self.valid_mask.shape)
        picam.set_controls({
            'AeEnable': False,
            'AwbEnable': False,
            'ExposureTime': self.camera_config.exposure_time,
            'AnalogueGain': self.camera_config.analogue_gain,
            'ColourGains': self.camera_config.colour_gains,
        })
        print(
            f"Capturing at {self.camera_config.size[0]}x{self.camera_config.size[1]} "
            f"@ target {self.camera_config.fps}fps."
        )
        return picam

    def update_clahe(self, clip_limit: float = None, tile_grid: Tuple[int, int] = None):
        """Apply live low-light tuning and rebuild the OpenCV CLAHE filter."""
        if clip_limit is not None:
            self.config.clahe_clip_limit = clip_limit
        if tile_grid is not None:
            self.config.clahe_tile_grid = tile_grid
        self._clahe = self._create_clahe()

    def _create_clahe(self) -> cv2.CLAHE:
        return cv2.createCLAHE(
            clipLimit=self.config.clahe_clip_limit,
            tileGridSize=self.config.clahe_tile_grid,
        )

    def process_frame(self, picam) -> Tuple[np.ndarray, Optional[Tuple[int, int]], Optional[Tuple[int, int]], Optional[np.ndarray]]:
        """Capture one frame and return ``(frame, ball_offset, goal_offset)``.

        Both targets are detected on every frame from the same capture. The ball
        uses the saturation-boosted HSV image its calibration was tuned against;
        the goal uses the camera HSV values unchanged. An offset is ``None`` when
        that target is not found.
        """
        frame = picam.capture_array()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

        original_frame = np.copy(frame)
        
        height, width = frame.shape[:2]
        centre_x, centre_y = width // 2, height // 2
        centre = (centre_x, centre_y)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        h, s, v = cv2.split(hsv)
        s_boosted = np.clip(s * 1.8, 0, 255).astype(np.uint8)
        hsv_ball = cv2.merge([h, s_boosted, v])

        if self.valid_mask.shape[:2] != hsv.shape[:2]:
            self.valid_mask = cv2.resize(
                self.valid_mask,
                (hsv.shape[1], hsv.shape[0]),  # (width, height)
                interpolation=cv2.INTER_NEAREST,
            )

        goal_target = self.config.target_goal
        ball_mask = self._masked_threshold(hsv_ball, 'ball')
        goal_mask = self._masked_threshold(hsv, goal_target)

        ball = self.find_ball(self._contours(ball_mask), self.config.min_contour_area)
        goal = self.find_goal(self._contours(goal_mask), self.config.goal_min_contour_area)

        cv2.circle(frame, centre, 4, (0, 255, 0), -1)
        
        cv2.circle(frame, centre, self.config.ball_dribble_radius, (255, 0, 0), 1)

        cv2.circle(frame, centre, self.config.orbit_radius, (0, 0, 255), 1)

        ball_offset = self._annotate(frame, ball, centre_x, centre_y, (0, 0, 255), (255, 0, 0))
        goal_offset = self._annotate(frame, goal, centre_x, centre_y, (0, 255, 255), (0, 200, 255))

        if self.config.debug_mask:
            ball_layer = frame.copy()
            ball_layer[ball_mask > 0] = (0, 255, 0)
            goal_layer = frame.copy()
            goal_layer[goal_mask > 0] = (255, 0, 0)
            frame = cv2.addWeighted(frame, 0.7, ball_layer, 0.3, 0)
            frame = cv2.addWeighted(frame, 0.7, goal_layer, 0.3, 0)

        return frame, ball_offset, goal_offset, original_frame

    def _masked_threshold(self, hsv: np.ndarray, target: str) -> np.ndarray:
        """Threshold ``hsv`` for ``target`` and restrict it to the valid region."""
        lower, upper = self._target_bounds(target)
        return cv2.bitwise_and(cv2.inRange(hsv, lower, upper), self.valid_mask)

    @staticmethod
    def _contours(mask: np.ndarray) -> list[np.ndarray]:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours

    @staticmethod
    def _annotate(frame, position, centre_x, centre_y, dot_colour, line_colour) -> Optional[Tuple[int, int]]:
        """Draw a target marker + centre line and return its offset from centre."""
        if position is None:
            return None
        x, y = position
        cv2.circle(frame, (x, y), 5, dot_colour, -1)
        cv2.line(frame, (centre_x, centre_y), (x, y), line_colour, 2)
        return (x - centre_x, y - centre_y)

    def _target_bounds(self, target: str) -> Tuple[np.ndarray, np.ndarray]:
        """Return the configured HSV bounds for a detection target."""
        bounds = {
            'ball': (self.config.ball_lower, self.config.ball_upper),
            'yellow_goal': (self.config.yellow_goal_lower, self.config.yellow_goal_upper),
            'blue_goal': (self.config.blue_goal_lower, self.config.blue_goal_upper),
        }
        try:
            lower_values, upper_values = bounds[target]
        except KeyError as exc:
            raise ValueError(
                f"Unknown vision target {target!r}; expected ball, yellow_goal, or blue_goal"
            ) from exc

        lower = np.array(lower_values)
        upper = np.array(upper_values)
        return lower, upper

    @staticmethod
    def find_ball(contours, min_area: int) -> Optional[Tuple[int, int]]:
        """Return the largest qualifying contour centroid, if one exists."""
        for contour in sorted(contours, key=cv2.contourArea, reverse=True):
            if cv2.contourArea(contour) <= min_area:
                break
            moments = cv2.moments(contour)
            if moments['m00'] == 0:
                continue
            ball_x = int(moments['m10'] / moments['m00'])
            ball_y = int(moments['m01'] / moments['m00'])
            return ball_x, ball_y
        return None
    
    @staticmethod
    def find_goal(contours, min_area: int) -> Optional[Tuple[int, int]]:
        """Return the largest qualifying contour centroid, if one exists."""
        for contour in sorted(contours, key=cv2.contourArea, reverse=True):
            if cv2.contourArea(contour) <= min_area:
                break
            moments = cv2.moments(contour)
            if moments['m00'] == 0:
                continue
            goal_x = int(moments['m10'] / moments['m00'])
            goal_y = int(moments['m01'] / moments['m00'])
            return goal_x, goal_y
        return None