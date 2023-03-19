import cv2
import os
from os import listdir

#Download the required video and place it in the same directory as this and provide the filename

filename = 'Introduction to Python for Data Science.mp4'
filename2 = 'frames4' # filename for storing frames

# Open video using OpenCV
cap = cv2.VideoCapture(filename)

# Set frame count and sampling rate
num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
sample_rate = 1000  # extract every 10th frame

print(num_frames)

ret, frame = cap.read()
# cv2.imshow('', frame)
# cv2.waitKey(0)

# ensure that no directory with this name already exists
os.mkdir(filename2)
for i in range(num_frames):
    ret, frame = cap.read()
    if i % sample_rate == 0:
        cv2.imwrite(f"{filename2}/frame_{i}.jpg", frame)

# Release video and close all windows
cap.release()
cv2.destroyAllWindows()
