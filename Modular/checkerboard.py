import cv2

img = cv2.imread("Photos/calib_29.png")  # BLANK 1: test this on calib_21.png or calib_22.png — the FULL, uncropped frame
if img is None:
    raise FileNotFoundError("imread returned None — check the path")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# crop is gray[y_start:y_end, x_start:x_end] — rows (height) first, then columns (width)
# board_crop = gray[90:170, 100:400]
# print("Board region min:", board_crop.min())
# print("Board region max:", board_crop.max())

# cv2.imwrite("board_crop_check.png", board_crop)  # save it so you can eyeball if you got the right region

found, corners = cv2.findChessboardCorners(
    gray,
    (3, 3),  # BLANK 2: back to your real pattern size, not (3,3) — that was only for the crop diagnostic
    flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
)

print("Found:", found)
if found:
    print("Number of corners detected:", len(corners))