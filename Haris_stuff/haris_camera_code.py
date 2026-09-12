# camera_calibrate.py
import json
import time
import numpy as np
import cv2
from picamera2 import Picamera2

SETTLE_SECONDS = 3
SAMPLE_COUNT = 15
BALL_PATCH_SIZE = 10
GOAL_PATCH_SIZE = 40

HUE_MARGIN = 8
SAT_MARGIN = 80
VAL_MARGIN = 100


def settle_and_lock_exposure(picamera):
    """
    Locks exposure, colour differences and other camera controls to
    ensure that the camera doesn't adjust camera controls when in movement
    (because then the HSV range will be useless if it does adjust)
    """
    print(f"Settling auto-exposure/white-balance for {SETTLE_SECONDS}s... "
          f"point the camera at your real match lighting now.")

    last_metadata = None
    start = time.time()
    while time.time() - start < SETTLE_SECONDS:
        picamera.capture_array()  # just to keep frames flowing
        last_metadata = picamera.capture_metadata()
        # 5 iterations per second is chosen to not waste processing power
        # while maintaining similar capture results.
        time.sleep(0.2)

    exposure_time = last_metadata.get("ExposureTime")
    analogue_gain = last_metadata.get("AnalogueGain")
    colour_gains = last_metadata.get("ColourGains")

    picamera.set_controls({
        "AeEnable": False,
        "AwbEnable": False,
        "ExposureTime": exposure_time,
        "AnalogueGain": analogue_gain,
        "ColourGains": colour_gains,
    })

    print(f"Locked: ExposureTime={exposure_time}  AnalogueGain={round(analogue_gain, 2)}  "
          f"ColourGains=({round(colour_gains[0], 2)}, {round(colour_gains[1], 2)})\n")

    return exposure_time, analogue_gain, colour_gains


def sample_ball_hsv(picamera):
    """
    Builds a capture zone of 10x10 pixels in the centre of the camera,
    finds the median HSV value of the 100 pixels and takes the median of
    15 of those HSV values to represent ball colour.
    """
    input("Hold the ball steady at the centre of the camera's view, then press Enter...")

    hue_samples, saturation_samples, value_samples = [], [], []

    for count in range(SAMPLE_COUNT):
        frame = picamera.capture_array()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

        height, width = frame.shape[:2]
        cx, cy = width // 2, height // 2
        ball_half = BALL_PATCH_SIZE // 2

        ball_patch = frame[cy - ball_half: cy + ball_half, cx - ball_half: cx + ball_half]
        hsv_patch = cv2.cvtColor(ball_patch, cv2.COLOR_BGR2HSV)

        hue_samples.append(np.median(hsv_patch[:, :, 0]))
        saturation_samples.append(np.median(hsv_patch[:, :, 1]))
        value_samples.append(np.median(hsv_patch[:, :, 2]))

        time.sleep(0.1)

    hue_median = float(np.median(hue_samples))
    saturation_median = float(np.median(saturation_samples))
    value_median = float(np.median(value_samples))

    print(f"Sampled ball HSV (median of {SAMPLE_COUNT} frames): "
          f"H={round(hue_median, 1)}  S={round(saturation_median, 1)}  V={round(value_median, 1)}\n")

    return hue_median, saturation_median, value_median


def sample_goal_hsv(picamera, goal_colour='yellow'):
    input(f"Point the camera at {goal_colour} goal, and then press Enter...")

    hue_samples, saturation_samples, value_samples = [], [], []

    for count in range(SAMPLE_COUNT):
        frame = picamera.capture_array()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

        height, width = frame.shape[:2]
        yellow_cx, yellow_cy = (width // 2,
                                height // 2 + (height // 4))
        goal_half = GOAL_PATCH_SIZE // 2

        yellow_goal_patch = frame[yellow_cy - goal_half: yellow_cy + goal_half,
                            yellow_cx - goal_half: yellow_cx + goal_half]
        hsv_patch = cv2.cvtColor(yellow_goal_patch, cv2.COLOR_BGR2HSV)

        hue_samples.append(np.median(hsv_patch[:, :, 0]))
        saturation_samples.append(np.median(hsv_patch[:, :, 1]))
        value_samples.append(np.median(hsv_patch[:, :, 2]))

        time.sleep(0.1)

    hue_median = float(np.median(hue_samples))
    saturation_median = float(np.median(saturation_samples))
    value_median = float(np.median(value_samples))

    print(f"Sampled goal HSV (median of {SAMPLE_COUNT} frames): "
          f"H={round(hue_median, 1)}  S={round(saturation_median, 1)}  V={round(value_median, 1)}\n")

    return hue_median, saturation_median, value_median


def build_range_ball(h_med, s_med, v_med):
    lower = [
        max(0, h_med - HUE_MARGIN),
        max(0, s_med - SAT_MARGIN),
        max(0, v_med - VAL_MARGIN),
    ]
    upper = [
        min(179, h_med + HUE_MARGIN),
        255,
        255,
    ]
    return lower, upper


def build_range_goal(h_med, s_med, v_med):
    lower = [
        max(0, h_med - 5),
        max(0, s_med - 40),
        max(0, v_med - 40),
    ]
    upper = [
        min(179, h_med + 5),
        min(255, s_med + 40),
        min(255, v_med + 40),
    ]
    return lower, upper


def main():
    picamera = Picamera2()
    picamera.configure(picamera.create_preview_configuration())
    picamera.start()
    time.sleep(1)  # let the sensor warm up before reading metadata

    exposure_time, analogue_gain, colour_gains = settle_and_lock_exposure(picamera)
    ball_hue_median, ball_saturation_median, ball_value_med = sample_ball_hsv(picamera)
    lower_orange, upper_orange = build_range_ball(ball_hue_median, ball_saturation_median, ball_value_med)

    yellow_hue_median, yellow_saturation_median, yellow_value_median = sample_goal_hsv(picamera, 'yellow')
    lower_yellow, upper_yellow = build_range_goal(yellow_hue_median, yellow_saturation_median, yellow_value_median)

    blue_hue_middle, blue_saturation_middle, blue_value_middle = sample_goal_hsv(picamera, 'blue')
    lower_blue, upper_blue = build_range_goal(blue_hue_middle, blue_saturation_middle, blue_value_middle)

    calibration = {
        "exposure_time": exposure_time,
        "analogue_gain": analogue_gain,
        "colour_gains": list(colour_gains),

        "lower_orange": lower_orange,
        "upper_orange": upper_orange,

        "lower_yellow": lower_yellow,
        "upper_yellow": upper_yellow,

        "lower_blue": lower_blue,
        "upper_blue": upper_blue,
    }

    with open("calibration.json", "w") as f:
        json.dump(calibration, f, indent=2)

    print("Saved calibration.json:")
    print(json.dumps(calibration, indent=2))


if __name__ == "__main__":
    main()