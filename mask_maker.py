import cv2
import numpy as np

ref = cv2.imread("reference_frame.png")
gray = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)

# Anything below this brightness is "self" / dead zone (black robot body + outer rim)
_, valid_mask = cv2.threshold(gray, 25, 255, cv2.THRESH_BINARY)

# Clean up small speckles/holes from noise
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
valid_mask = cv2.morphologyEx(valid_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
valid_mask = cv2.morphologyEx(valid_mask, cv2.MORPH_OPEN, kernel, iterations=2)

cv2.imwrite("valid_region_mask.png", valid_mask)