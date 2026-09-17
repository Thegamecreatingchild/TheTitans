from picamera2 import Picamera2
import cv2
import time

picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"size": (640, 480)}))
picam2.start()
time.sleep(2)  # let AE/AWB converge

frame = picam2.capture_array()
print(frame.shape, frame.dtype)  # sanity check — expect (868, 982, 4) uint8

# XBGR8888 -> drop the X (alpha/pad) channel and convert to standard BGR
frame_bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)

cv2.imwrite("reference_frame.png", frame_bgr)