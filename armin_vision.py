"""
Author: Aditya Gantimahapatruni
Date created: 18/9/2026

File purpose: TLDR - A Camera Handler

Description:
This file contains a class that handles most vision logic, including boosting saturation of orange pixels,
finding a ball, and returning its position to allow other modules to operate with new info.

"""

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
        self.valid_mask = cv2.imread("valid_region_mask.png", cv2.IMREAD_GRAYSCALE)
        if self.valid_mask is None:
            raise FileNotFoundError(
                "valid_region_mask.png not found or unreadable — "
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
        # debug_config = picam.create_still_configuration()
        # width, height = debug_config["main"]["size"]
        # print(width, height)
        
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
        h, s, v = cv2.split(hsv)
        sat_gain = 1.8
        s = np.clip(s * sat_gain, 0, 255).astype(np.uint8)
        hsv = cv2.merge([h, s, v])

        lower = np.array([self.config.h_low, self.config.s_low, self.config.v_low])
        upper = np.array([self.config.h_high, self.config.s_high, self.config.v_high])
        mask = cv2.inRange(hsv, lower, upper)

        if self.valid_mask.shape[:2] != mask.shape[:2]:
            self.valid_mask = cv2.resize(
                self.valid_mask,
                (mask.shape[1], mask.shape[0]),  # (width, height)
                interpolation=cv2.INTER_NEAREST,
            )

        mask = cv2.bitwise_and(mask, self.valid_mask)

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
        
        if self.config.debug_mask:
            return cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), offset
        
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
