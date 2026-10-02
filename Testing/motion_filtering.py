import cv2, numpy as np

def filter_motion():
    video = cv2.VideoCapture(r'Testing\MHS First RoboCup 2014.mp4')
    subtractor = cv2.createBackgroundSubtractorMOG2(20, 50)

    while True:
        ret, frame = video.read()
        
        if ret:
            mask = subtractor.apply(frame)
            cv2.imshow('mask', mask)
            
            if cv2.waitKey(5) == ord('X'):
                break

        else:
            video = cv2.VideoCapture(r'Testing\MHS First RoboCup 2014.mp4')
        
        
    cv2.destroyAllWindows()
    video.release()