# USAGE
# python detect.py --input path_to_video

# import the necessary packages
from __future__ import print_function
from imutils.object_detection import non_max_suppression
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True, help="path to images directory")
args = vars(ap.parse_args())

# initialize the HOG descriptor/person detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# capture video
cap = cv2.VideoCapture(args['input'])

# read frame for select region of interest
ret, frame_for_roi = cap.read()
frame_for_roi = imutils.resize(frame_for_roi, width=min(1000, frame_for_roi.shape[1]))

# select ROI
r = cv2.selectROI("Please select ROI", frame_for_roi, ret)

while(True):

    # Capture frame-by-frame
    ret, frame = cap.read()

    # Video end checker
    if not ret:
        print('video has reached the end')
        break

    # Resize video
    frame = imutils.resize(frame, width=min(1000, frame.shape[1]))

    # Draw roi on frame
    cv2.rectangle(frame, (r[0], r[1]), (r[0] + r[2], r[1] + r[3]), (255, 0, 0), 1)

    # Detect people in the roi
    (rects, weights) = hog.detectMultiScale(frame[r[1]:r[1] + r[3], r[0]:r[0] + r[2]], winStride=(4, 4),
                                            padding=(8, 8), scale=1.05)

    # Apply non-maxima suppression to the bounding boxes using a
    # fairly large overlap threshold to try to maintain overlapping
    # boxes that are still people
    rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
    pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)

    # Draw the final bounding boxes
    for (xA, yA, xB, yB) in pick:
        cv2.rectangle(frame, (r[0] + xA, r[1] + yA), (r[0] + xB, r[1] + yB), (0, 255, 0), 2)

    # Display counter
    cv2.putText(frame, 'detected = ' + str(len(pick)), (20, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('frame',frame)

    # Kill script by press 'Q' button
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()