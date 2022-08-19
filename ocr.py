import cv2 
import matplotlib.pyplot as plt
import pytesseract

image = cv2.imread('tmp.jpg')

resizeImg = cv2.resize(image,None,fx=2,fy=2,interpolation=cv2.INTER_CUBIC)
finalImg = cv2.GaussianBlur(resizeImg,(5,5),0)
result = pytesseract.image_to_string(finalImg,lang ='eng',
config ='--oem 3 -l eng --psm 6')
print(result)

# plt.figure(figsize=(20,20))
plt.title("Original")
plt.imshow(finalImg)