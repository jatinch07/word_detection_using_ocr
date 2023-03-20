# Word_detection_using_ocr

## video.py
This file takes a video input and returns the .jpg slides obtained from the video

1. Download the desired video and put it in the same directory as the python project
2. Type the folder name you want to save the .jpg slides into
3. Ensure that no directory with the given name already exists
4. Run the code to obtain the folder containing .jpg slides.

## tesseract.py
This file takes a .jpg input and returns the text contained inside the photo

1. Enter the directory location where pytesseract is installed in your device
2. Put the desired image in the same directory as the python project
3. Enter the .jpg image name at line 7
4. Run the code to obtain the results (text)

## main.py

This file takes a .jpg image and returns the different rectangular contours obtained in the image, these outputs are also in the .jpg form, which can be given as input to the tesseract library (as explained in the above section) to obtain text output

1. Put the desired image in the same directory as the python project
2. Enter the .jpg image name in line 67
3. Increase or decrease the kernel size in line 72 and 77 to change the size of the desired rectangular contours
4. All the contours are saved in the list 'save_contours'
5. These contours can be given as input to the tesseract library as shown in the above section

(video.py has been merged into main.py to automate the process, that part can be avoided for testing using cntl+/, iterate through the .jpg files obtained from video.py to get the contours from all slides)
