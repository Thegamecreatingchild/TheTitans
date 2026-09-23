"""
One script version of the bot's code. This is to make code simpler and easier to read.
"""

import asyncio
import math
import signal
import sys
import time
import numpy as np
from dataclasses import dataclass
from typing import Iterable, Set, Tuple
from steelbar_powerful_bldc_driver import PowerfulBLDCDriver

from picamera2 import Picamera2, Preview
import board
import busio
import cv2
import websockets
from gpiozero import Button

# An easy way to store data - it is like a record or a struct in other languages. It is used to store calibration values for motors.
# Accessing anything from inside is as simple as calling calibration.elecangleoffset or calibration.sincoscentre.
@dataclass
class Calibration:
    """
    Stores calibration values for a motor, easily accessible by calling
    calibration.elecangleoffset and calibration.sincoscentre. 
    These values are used to calibrate the motor's position and speed.
    """
    elecangleoffset: int
    sincoscentre: int


class Motor:
    def __init__(self, i2c, address, calibration):
        self.motor = PowerfulBLDCDriver(i2c, address)
        self.motor.set_current_limit_foc(65536 * 2) # Set the current limit for the motor
        self.motor.set_id_pid_constants(1500, 200) # Set the PID constants for current control
        self.motor.set_speed_pid_constants(4e-2, 4e-4, 3e-2) # Set the PID constants for speed control
        self.motor.set_ELECANGLEOFFSET(calibration.elecangleoffset) # Set the ELECANGLEOFFSET to calibration value
        self.motor.set_SINCOSCENTRE(calibration.sincoscentre) # Set the SINCOSCENTRE to calibration value
        self.motor.configure_operating_mode_and_sensor(3, 1) # Set to position control mode
        self.motor.configure_command_mode(12) # Set to speed control mode
        self.motor.set_speed(0)
    
    def set_speed(self, speed: int):
        """Set the speed of the motor. Speed must be within range of max_speed."""
        self.motor.set_speed(speed)

class Robot:
    def __init__(self):
        self.motor_config()
        self.vision_config()
        self.camera_config()
        self.manual_controls_config()
    
    def motor_config(self):
        """Configuring evreything to do with motors / movement logic, including the dribbler."""
        self.motors : list[Motor] = []
        
        self.adresses = (26, 28, 27, 25)
        self.calibrations: Tuple[Calibration, ...] = (
        Calibration(1327731200, 1241), # 26
        Calibration(1435147520, 1243), # 28
        Calibration(1256835584, 1258), # 27
        Calibration(1150337792, 1247), # 25
        )
        self.max_speed: int = 100_000_000
        
        # Dribbler stuff
        self.dribbler_enabled: bool = False
        self.dribbler_address: int = 29
        self.dribbler_calibration: Calibration = Calibration(1437511680, 1245)
        self.dribbler_max_speed: float = self.max_speed * 1.5
        
        # Movement stuff
        self.orbit_standoff_radius : int = 170
        self.orbit_arrived_angle_tolerance : int = 10
        self.orbit_radial_gain : int = 400_000 
        # ! The gain for the radial component of orbiting. Requires tuning. 
        # Higher values make the robot orbit closer to the target.
        # Lower values make the robot orbit further from the target.
    
    def vision_config(self):
        """Configuring everything to do with vision, including ball and goal detection."""
        self.ball_lower: Tuple[int] = (0, 200, 77)
        self.ball_upper: Tuple[int] = (15, 255, 255)
        
        self.ball_min_contour_area: int = 1
        
        self.dribble_trigger_radius: int = 140
        self.orbit_trigger_radius: int = 200
        self.camera_rotation_offset: float = 90.0
        
        self.ball_offset : Tuple[int] = (0, 0)
        self.ball_lost_timeout: float = 0.3
        
        self.debug_mask : bool = False
        
        self.yellow_goal_lower: Tuple[int] = (20, 235, 100)
        self.yellow_goal_upper: Tuple[int] = (40, 255, 255)
        
        self.blue_goal_lower: Tuple[int] = (95, 207, 60)
        self.blue_goal_upper: Tuple[int] = (105, 255, 100)
    
        self.goal_stop_distance : int = 100
        self.goal_min_contour_area : int = 120  # goals are big; a larger floor rejects speckle
        self.target_goal : str = 'yellow_goal'  # 'yellow_goal' or 'blue_goal' - the goal we attack

        self.bot_mask = cv2.imread('bot_mask.png', cv2.IMREAD_GRAYSCALE)
        self.bot_mask = cv2.rotate(self.bot_mask, cv2.ROTATE_90_COUNTERCLOCKWISE)
        
        
        
    def camera_config(self):
        """Configuring everything to do with the camera, including websockets connection and camera settings."""
        self.camera_size: Tuple[int, int] = (640, 480)
        self.camera_fps: int = 120
        self.camera_exposure_time = 8333
        self.frame_interval_microseconds: int = 1000000 // self.camera_fps  # 120 fps
        self.analogue_gain: float = 8.677966117858887
        self.colour_gains: Tuple[float, float] = (2.4364535808563232, 1.9698092937469482)
    
    def manual_controls_config(self):
        """Configuring everything to do with manual controls, including timeouts and switch pins."""
        self.manual_keys: dict[str, int] = {'w': 0, 'd': 90, 's': 180, 'a': 270} # Bearing mappings to each key
        self.spin_keys: dict[str, int] = {'q': 1, 'e': -1}
        
        self.miscellaneous_keys: dict[str, str] = {'k': 'dribble', 'x': 'orbit'}
        
        self.key_lost_timeout: float = 0.3
        self.motor_loop_delay: float = 0.02
        self.camera_loop_delay: float = 0.03
        
        self.movement_switch_gpio_pin: int = 26

    # -- Motor control methods - -

    def setup_motors(self):
        self.i2c = busio.I2C(board.SCL, board.SDA)
        self.motors = []
        for address, calibration in zip(self.adresses, self.calibrations):
            motor = Motor(self.i2c, address, calibration)
            self.motors.append(motor)
        
        if self.dribbler_enabled:
            self.dribbler = Motor(self.i2c, self.dribbler_address, self.dribbler_calibration)
        else:
            self.dribbler = None
        
        print("All motors ready.")
    
    def move(self, degree: float, speed: int = None) -> None:
        """Move the bot in a direction, commanding each wheel using math. Check the OneNote to see it."""
        
        if speed is None: speed = self.max_speed
        
        angle_rad = math.radians(degree + 90)
        
        # Motor drivers require integer values (hence the rounding)
        x = math.floor(math.cos(angle_rad) * speed)
        y = math.floor(math.sin(angle_rad) * speed)
        self.motors[0].set_speed(y + x)
        self.motors[1].set_speed(y - x)
        self.motors[2].set_speed(-(y + x))
        self.motors[3].set_speed(-(y - x))
    
    def stop(self) -> None:
        """Stop the bot by setting all motors to 0 speed."""
        for motor in self.motors: motor.set_speed(0)
        if self.dribbler_enabled:
            self.dribbler.set_speed(0)
        
    def spin(self, speed: int) -> None:
        """Spin the bot in place by setting all motors to the same speed."""
        for motor in self.motors: motor.set_speed(speed)
    
    def orbit_to_dribbling_range(self, target_bearing: float = 0.0) -> bool:
        """Orbit around the ball to the specified bearing."""
        
        offset = self.ball_offset
        if offset is None: # No ball detected
            self.stop()
            return False
        
        dx, dy = offset
        ball_distance = math.hypot(dx, dy)
        ball_bearing = (
            math.degrees(math.atan2(dx, -dy)) + self.camera_rotation_offset
        ) # uses tan^-1 to get bearing, also adding camera offset.
        
        normalized_ball_bearing = (ball_bearing - target_bearing + 180) % 360 - 180
        # This turns the [0, 360) into [0, 180). Easier to choose which way to go.
        # As it gets close to 0, the bot is closer to being at the target bearing.
        
        if (abs(normalized_ball_bearing) <= self.orbit_arrived_angle_tolerance):
            if (abs(ball_distance - self.orbit_standoff_radius) <= self.orbit_standoff_radius * 0.15):
                self.stop() # ! Please change to make the bot move to goal next.
                return True # The bot has arrived at the ball.
            self.move(offset, self.max_speed * 0.5) # makes sure it is within 
            return False
        
        # Follows tangent to the ball.
        tangent_dir = ball_bearing + (90 if normalized_ball_bearing > 0 else -90)
        # Lower when normalized ball bearing is lower, higher when it's higher.
        ease = min(abs(normalized_ball_bearing) / 45, 1)
        tangent_speed = self.max_speed * ease
        
        # If the bot is too close to the ball.
        proximity_error = ball_distance - self.orbit_standoff_radius
        # How big the resulting vector should be that allows it to move away from the ball.
        proximity_speed = max(-self.max_speed * 0.5, min(self.max_speed * 0.5, proximity_error * self.orbit_radial_gain))
        
        # Vector addition to get final speed and direction.
        # Vector t for tangent, vector p for proximity, vector v for final vector.
        
        # Conversion into vectors components
        tdx = math.sin(math.radians(tangent_dir)) # We are using sin on x and cos on y because we are using Polar
        tdy = -math.cos(math.radians(tangent_dir)) # Angle is measured from y-axis, therefore we use sin on x and cos on y.
        pdx = math.sin(math.radians(ball_bearing)) # Imagine rotating the cartesian plane 90 degrees clockwise. That is how we are measuring the angle
        pdy = -math.cos(math.radians(ball_bearing)) # so that sin and cos work for us. The negative is because the y-axis is inverted in the camera (not for negative cos).
        
        # Adding the magnitudes of the vectors before summing the components.
        vx = tangent_speed * tdx + proximity_speed * pdx
        vy = tangent_speed * tdy + proximity_speed * pdy

        # Keeping final bearing in [0, 360) and final speed within max_speed.
        final_bearing = math.degrees(math.atan2(vx, -vy)) % 360
        final_speed = int(min(math.hypot(vx, vy), self.max_speed))
        self.move(final_bearing, final_speed)
        return False # These statements represent if the bot has not arrived at the ball yet, and is still orbiting to get there.
    
    # -- Vision methods - -
    
    def setup_camera(self):
        """Setup the camera and return the picamera2 object."""
        picam = Picamera2()
        config = picam.create_preview_configuration(
            main={"size": self.camera_size, "format": "RGB888"},
            lores={"size": (320, 240), "format": "YUV420"},
            controls={
                "FrameDurationLimits": (
                    self.frame_interval_microseconds,
                    self.frame_interval_microseconds,
                )
            },
        )
        picam.configure(config)
        
        picam.start()
        # Setting the camera's fps to be higher by reducing the exposure time and various others.
        picam.set_controls({
            'AeEnable': False,
            'AwbEnable': False,
            'ExposureTime': self.camera_exposure_time,
            'AnalogueGain': self.analogue_gain,
            'ColourGains': self.colour_gains,
        })
        print(
            f"Capturing at {self.camera_size[0]}x{self.camera_size[1]} "
            f"@ target {self.camera_fps}fps."
        )
        return picam
    
    def masked_threshold(self, hsv: np.ndarray, target: str) -> np.ndarray: 
        targets = {
            'ball': (self.ball_lower, self.ball_upper),
            'blue_goal': (self.blue_goal_lower, self.blue_goal_upper),
            'yellow_goal': (self.yellow_goal_lower, self.yellow_goal_upper),
        }
        
        try:
            lower, upper = targets[target]
        except KeyError:
            raise ValueError(f"Unknown target: {target}. Valid targets are: {list(targets.keys())}")
        
        return cv2.bitwise_and(cv2.inRange(hsv, lower, upper), self.bot_mask)

    def contours(mask: np.ndarray):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours

    def annotate(frame, position, centre_x, centre_y, dot_colour, line_colour):
        """Draw a target marker + centre line and return its offset from centre."""
        if position is None: return None
        x, y = position
        cv2.circle(frame, (x, y), 5, dot_colour, -1)
        cv2.line(frame, (centre_x, centre_y), (x, y), line_colour, 2)
        return (x - centre_x, y - centre_y)

    def find_ball(contours, min_area: int):
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
    
    def find_goal(contours, min_area: int):
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

    def process_frame(self, picam):
        frame = picam.capture_array()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE) # Turns so camera is upright.

        height, width = frame.shape[:2] # 
        centre_x, centre_y = width // 2, height // 2
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) # hsv is now the frame

        h, s, v = cv2.split(hsv)
        s_boosted = np.clip(s * 1.8, 0, 255).astype(np.uint8) # Boosting saturation for better detection.
        hsv_ball = cv2.merge([h, s_boosted, v])

        if self.bot_mask.shape[:2] != hsv.shape[:2]:
            self.bot_mask = cv2.resize(
                self.bot_mask,
                (hsv.shape[1], hsv.shape[0]),  # (width, height)
                interpolation=cv2.INTER_NEAREST,
            )

        goal_target: str = self.target_goal
        ball_mask = self.masked_threshold(hsv_ball, 'ball')
        goal_mask = self.masked_threshold(hsv, goal_target)

        ball = self.find_ball(self.contours(ball_mask), self.ball_min_contour_area)
        goal = self.find_goal(self.contours(goal_mask), self.goal_min_contour_area)

        cv2.circle(frame, (centre_x, centre_y), 4, (0, 255, 0), -1)

        ball_offset = self.annotate(frame, ball, centre_x, centre_y, (0, 0, 255), (255, 0, 0))
        goal_offset = self.annotate(frame, goal, centre_x, centre_y, (0, 255, 255), (0, 200, 255))

        if self.debug_mask: # Highlights the visible contours to allow for better debugging at runtime.
            ball_layer = frame.copy()
            ball_layer[ball_mask > 0] = (0, 255, 0)
            goal_layer = frame.copy()
            goal_layer[goal_mask > 0] = (255, 0, 255)
            frame = cv2.addWeighted(frame, 0.7, ball_layer, 0.3, 0)
            frame = cv2.addWeighted(frame, 0.7, goal_layer, 0.3, 0)
        
        return frame, ball_offset, goal_offset