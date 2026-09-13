"""Camera setup and ball detection for Hari's robot.

Frames are rotated counter-clockwise before detection because of the camera's
physical mounting. The returned offset is ``(dx, dy)`` from image centre.
"""

import json
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
        self._clahe = self._create_clahe()

    def setup_camera(self) -> Picamera2:
        """Start the camera and apply saved exposure/white-balance settings."""
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

        try:
            with open('calibration.json') as file:
                calibration = json.load(file)
            picam.set_controls({
                'AeEnable': False,
                'AwbEnable': False,
                'ExposureTime': calibration['exposure_time'],
                'AnalogueGain': calibration['analogue_gain'],
                'ColourGains': tuple(calibration['colour_gains']),
            })
            print(
                f"Loaded calibration.json. Capturing at "
                f"{self.camera_config.size[0]}x{self.camera_config.size[1]} "
                f"@ target {self.camera_config.fps}fps."
            )
        except FileNotFoundError:
            print(
                f"No calibration.json found. Capturing at "
                f"{self.camera_config.size[0]}x{self.camera_config.size[1]} "
                f"@ target {self.camera_config.fps}fps."
            )
        return picam

    def update_clahe(self, clip_limit: float = None,
                     tile_grid: Tuple[int, int] = None) -> None:
        """Apply live low-light tuning and rebuild the OpenCV CLAHE filter."""
        if clip_limit is not None:
            self.config.clahe_clip_limit = clip_limit
        if tile_grid is not None:
            self.config.clahe_tile_grid = tile_grid
        self._clahe = self._create_clahe()

    def _create_clahe(self):
        return cv2.createCLAHE(
            clipLimit=self.config.clahe_clip_limit,
            tileGridSize=self.config.clahe_tile_grid,
        )

    def process_frame(self, picam) -> Tuple[np.ndarray, Optional[Tuple[int, int]]]:
        """Capture, threshold, select, and annotate one frame."""
        frame = picam.capture_array()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

        height, width = frame.shape[:2]
        centre_x, centre_y = width // 2, height // 2
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h, s, value = cv2.split(hsv)
        value = self._clahe.apply(value)
        hsv = cv2.merge([h, s, value])

        lower = np.array([self.config.h_low, self.config.s_low, self.config.v_low])
        upper = np.array([self.config.h_high, self.config.s_high, self.config.v_high])
        mask = cv2.inRange(hsv, lower, upper)
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        ball = self.find_ball(
            contours,
            centre_x,
            centre_y,
            self.config.min_contour_area,
            self.config.dead_zone_radius,
        )

        cv2.circle(frame, (centre_x, centre_y), 4, (0, 255, 0), -1)
        cv2.circle(
            frame,
            (centre_x, centre_y),
            int(self.config.dead_zone_radius),
            (128, 128, 128),
            2,
        )

        offset = None
        if ball is not None:
            ball_x, ball_y = ball
            offset = (ball_x - centre_x, ball_y - centre_y)
            cv2.circle(frame, (ball_x, ball_y), 5, (0, 0, 255), -1)
            cv2.line(frame, (centre_x, centre_y), ball, (255, 0, 0), 2)
        return frame, offset

    @staticmethod
    def find_ball(contours, centre_x: int, centre_y: int,
                  min_area: int, ignore_radius: int):
        """Return the largest qualifying contour centroid, if one exists."""
        for contour in sorted(contours, key=cv2.contourArea, reverse=True):
            if cv2.contourArea(contour) <= min_area:
                break
            moments = cv2.moments(contour)
            if moments['m00'] == 0:
                continue
            ball_x = int(moments['m10'] / moments['m00'])
            ball_y = int(moments['m01'] / moments['m00'])
            if math.hypot(ball_x - centre_x, ball_y - centre_y) < ignore_radius:
                continue
            return ball_x, ball_y
        return None
