import cv2
import numpy as np
from picamera2 import Picamera2

# Set up the Pi camera instead of a USB webcam
picamera = Picamera2()
picamera.configure(picamera.create_preview_configuration())
picamera.start()

params = cv2.SimpleBlobDetector_Params()
params.filterByArea = False
params.filterByCircularity = True
params.minCircularity = 0.35
params.filterByInertia = False
params.filterByConvexity = False
params.filterByColor = True
params.blobColor = 255
detector = cv2.SimpleBlobDetector_create(params)

while True:
    # Picamera2 gives frames in RGB — convert to BGR so OpenCV colors behave
    frame = picamera.capture_array()
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    hsvFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) # Converts the frame from BGR to HSV color space

    lower_orange = np.array([5, 141, 120]) # Lower bound for orange color in HSV
    upper_orange = np.array([15, 255, 255]) # Upper bound for orange color in HSV

    mask = cv2.inRange(hsvFrame, lower_orange, upper_orange) # Creates a mask for the orange color

    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # Finds contours in the mask
    keypoints = detector.detect(mask)

    circular_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        if circularity > 0.65:
            circular_contours.append((contour, circularity))

    frame = cv2.drawKeypoints(
        frame,
        keypoints,
        None,
        (0, 255, 255),
        cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )
    cv2.putText(frame, f'blobs: {len(keypoints)}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    if keypoints:
        best_keypoint = max(keypoints, key=lambda kp: kp.size)
        kpX, kpY = int(best_keypoint.pt[0]), int(best_keypoint.pt[1])
        cv2.line(frame, (frame.shape[1]//2, frame.shape[0]//2), (kpX, kpY), (0, 0, 255), 2) # Draws a line from the center of the frame to the yellow blob center

    if circular_contours:
        biggest_contour, best_circularity = max(circular_contours, key=lambda c: cv2.contourArea(c[0]))
        contour_area = cv2.contourArea(biggest_contour)

        if contour_area > 0:
            moments = cv2.moments(biggest_contour)

            # m00 is the total area of the contour
            # m10 is the weighted x sum, m01 is the weighted y sum
            # dividing by m00 gives the average (centroid) x and y coordinates

            if moments['m00'] != 0: # Checks if the area is not zero to avoid division by zero
                cX = int(moments['m10'] / moments['m00']) # Calculates the x coordinate of the center of the contour
                cY = int(moments['m01'] / moments['m00']) # Calculates the y coordinate of the center of the contour
                cv2.circle(frame, (cX, cY), 5, (0, 255, 0), -1) # Draws a green circle at the center of the contour
                cv2.drawContours(frame, [biggest_contour], -1, (0, 255, 0), 2)
                cv2.putText(frame, f'circularity: {best_circularity:.2f}', (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            largest = max(contours, key=cv2.contourArea)

            M = cv2.moments(largest)
            ball_x = int(M["m10"] / M["m00"])
            ball_y = int(M["m01"] / M["m00"])

            # Frame dimensions
            frame_h, frame_w = frame.shape[:2]
            centre_x = frame_w // 2
            centre_y = frame_h // 2

            norm_x = (ball_x - centre_x) / centre_x
            norm_y = (ball_y - centre_y) / centre_y

            # Flip y — camera y increases downward, but forward for us is up in frame
            norm_y = -norm_y
            print(norm_x, norm_y)

    cv2.imshow('frame', frame)
    cv2.imshow('mask', mask) # Displays the mask in a window named 'mask'

    if cv2.waitKey(1) == ord('q'):
        break

picamera.stop()
cv2.destroyAllWindows() # Closes all OpenCV windows