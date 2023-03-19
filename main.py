import cv2
import numpy as np
import os
from os import listdir


array = np.zeros([500, 500, 3],
                 dtype=np.uint8)
array[:, :] = [255, 255, 255]

array1= np.zeros([500, 500, 3],
                 dtype=np.uint8)
array1[:, :] = [255, 255, 255]


def fory(elem):
    return elem[0][0][1]


def forx(elem):
    return elem[0][0][0]


def mod(a,b):
    if a-b >= 0:
        return a-b
    else:
        return b-a


def difference(a,b):
    return a-b

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

# ensure that no directory with this name already exists
os.mkdir(filename2)
for i in range(num_frames):
    ret, frame = cap.read()
    if i % sample_rate == 0:
        cv2.imwrite(f"{filename2}/frame_{i}.jpg", frame)

image_list = []
# folder location
folder_dir = "C:/Users/hp/PycharmProjects/opencv1/frames4"
for images in listdir(folder_dir):

    # check if the image ends with jpg
    if images.endswith(".jpg"):
        image_list.append(images)

#iterate though all the images in image
large = cv2.imread('slide2.jpg')
large = cv2.resize(large, (1008, 1008))
rgb = cv2.pyrDown(large)
small = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)) ##3,3 #2,7
grad = cv2.morphologyEx(small, cv2.MORPH_GRADIENT, kernel)

_, bw = cv2.threshold(grad, 0.0, 255.0, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,1)) ##3,1#1,5
connected = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)
bvb = connected

# using RETR_EXTERNAL instead of RETR_CCOMP
contours, hierarchy = cv2.findContours(connected.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
# For opencv 3+ comment the previous line and uncomment the following line
# _, contours, hierarchy = cv2.findContours(connected.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

mask = np.zeros(bw.shape, dtype=np.uint8)

i = 0
p = []

print(len(contours))
for cont in contours:
    x, y, w, h = cv2.boundingRect(cont)
    mask[y:y + h, x:x + w] = 0

    cv2.drawContours(mask, contours, i, (255, 255, 255), -1)
    i += 1

    r = float(cv2.countNonZero(mask[y:y + h, x:x + w])) / (w * h)

    if r > 0.3 and w > 5 and h > 7:
        p.append(cont)
        # pp.append([p[ii][0][0][0], p[ii][0][0][1]])
        # ii += 1


p.sort(key=fory)

count = 1
m = 0

while m < 2 * len(p):
    for obj in range(len(p)-1):
        if mod(p[obj][0][0][1], p[obj+count][0][0][1]) <= 25 and \
                        difference(p[obj][0][0][0], p[obj+count][0][0][0]) >= 0:
                    saved_obj = p[obj+count]
                    p.pop(obj+count)
                    p.insert(obj+count, p[obj])
                    p.pop(obj)
                    p.insert(obj, saved_obj)
    m += 1


xn, yn, wn, hn = 0, 0, 0, 0
a = 0
bb = 66
save_contours = []
print(len(p))
for idx in range(len(p)):
    x, y, w, h = cv2.boundingRect(p[idx])
    mask[y:y+h, x:x+w] = 0

    cv2.drawContours(mask, p[0:bb], idx, (255, 255, 255), -1)

    r = float(cv2.countNonZero(mask[y:y+h, x:x+w])) / (w * h)

    if r > 0.3 and w > 3 and h > 5:
        roi_color = rgb[y:y + h, x:x + w]
        save_contours.append(roi_color)
        print("[INFO] Object found. Saving locally.")


    if r > 0.3 and w > 3 and h > 5:
        cv2.rectangle(rgb, (x, y), (x+w-1, y+h-1), (0, 255, 0), 1)


print(save_contours)

cv2.imshow("roi", roi_color)

cv2.imshow('rects', rgb)
cv2.waitKey(0)
cv2.destroyAllWindows()
