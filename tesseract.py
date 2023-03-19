import pytesseract
import PIL.Image
from os import listdir

pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'

text = pytesseract.image_to_string(PIL.Image.open("image.png"))

print(text)
