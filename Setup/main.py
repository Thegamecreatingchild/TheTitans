"""
One script version of the bot's code. This is to make code simpler and easier to read.
"""

import asyncio
import math
import signal
import sys
import time

import cv2
import websockets
from gpiozero import Button

class Robot:
    def __init__(self):
        # Motor stuff
        self.motor_config()
    
    def motor_config(self):
        self.adresses = (26, 28, 27, 25)
        calibrations: tuple[tuple] = (
        (1327731200, 1241),
        (1435147520, 1243),
        (1256835584, 1258),
        (1150337792, 1247),
        )
        self.max_speed: int = 100_000_000
        self.enable_dribbler: bool = False
        self.dribbler_address: int = 29
        self.dribbler_calibration: tuple = (1437511680, 1245)
        self.dribbler_speed: int = self.max_speed * 1.5
    
    def vision_config(self):        
        self.ball_lower: tuple[int] = (0, 200, 77)
        self.ball_upper: tuple[int] = (15, 255, 255)
        
        self.ball_min_contour_area: int = 1
        
        self.ball_dribble_radius: int = 140
        self.orbit_radius: int = 200
        self.camera_rotation_offset: float = 90.0
        
        self.ball_lost_timeout: float = 0.3
        
        self.debug_mask : bool = False
        
        self.yellow_goal_lower: tuple[int] = (20, 235, 100)
        self.yellow_goal_upper: tuple[int] = (40, 255, 255)
        
        self.blue_goal_lower: tuple[int] = (95, 207, 60)
        self.blue_goal_upper: tuple[int] = (105, 255, 100)
    
        self.goal_stop_distance : int = 100
        self.goal_min_contour_area : int = 120  # goals are big; a larger floor rejects speckle
        self.target_goal : str = 'yellow_goal'  # 'yellow_goal' or 'blue_goal' - the goal we attack