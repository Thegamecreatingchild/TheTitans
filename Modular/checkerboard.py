import cv2
from pathlib import Path

dir = Path("Photos")

success = 0
total = 0

failed = []

for item in dir.iterdir():
    print(item.name)
    img = cv2.imread(f"Photos/{item.name}")  # BLANK 1: test this on calib_21.png or calib_22.png — the FULL, uncropped frame
    if img is None:
        raise FileNotFoundError("imread returned None — check the path")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    found, corners = cv2.findChessboardCorners(
        gray,
        (3, 3),  # BLANK 2: back to your real pattern size, not (3,3) — that was only for the crop diagnostic
        flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    )
    total += 1
    print("Found:", found)
    if found:
        success += 1
        print("Number of corners detected:", len(corners))
    else:
        failed.append(item.name)

print(sorted(failed))
print(float(success / total) * 100)