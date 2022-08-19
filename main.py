import matplotlib.pyplot as plt
from PIL import Image
import cv2 as cv
import os
import pytesseract

model = 'model/frozen_inference_graph.pb'
LABELS = ["null","Plat Nomor"]    
pbtxt = 'model/graph.pbtxt'
image = cv.imread('image2.jpg')
image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

hgt, wdt = image.shape[:2]
start_row = int(hgt * .50)
start_col = int(wdt * .50)
end_row = int(hgt * .95)
end_col = int(wdt * .95)
cropImage = image[start_row:end_row,start_col:end_col]

cvNetwork = cv.dnn.readNetFromTensorflow(model,pbtxt)
rows = image.shape[0]
cols = image.shape[1]
cvNetwork.setInput(cv.dnn.blobFromImage(image, size=(300, 300), crop=False))
cvOut = cvNetwork.forward()

# plt.colorbar(
plt.figure(figsize=(20,20))
plt.subplot(2,2,1)
plt.title("Original")
plt.imshow(image)


plt.figure(figsize=(20,20))
plt.subplot(2,2,2)
plt.title("Cropped")
filename = "tmp.jpg".format(os.getpid())
cv.imwrite(filename,cropImage)
plt.imshow(cropImage)


# targetImg = cv.cvtColor(cropImage,cv.COLOR_BGR2GRAY)
# txt = pytesseract.image_to_string(Image.open(filename))
# print("OCR "+txt)
# plt.figure(figsize=(20,20))
# plt.subplot(2,2,4)
# plt.title("OCR")
# plt.imshow(targetImg)


for detection in cvOut[0,0,:,:]:
    score = float(detection[2])
    if score > 0.5:
        label = "{}: {:.2f}%".format(LABELS[int(detection[1])], detection[2] * 100)
        print("[INFO] {}".format(label))
        left = detection[3] * cols
        top = detection[4] * rows
        right = detection[5] * cols
        bottom = detection[6] * rows

        filename = "tmp.jpg".format(os.getpid())
        cv.imwrite(filename,image[int(top):int(bottom)+10,int(left):int(right)+10])

        
        print("left:{} top:{} right:{} bottom:{}".format(left,top,right,bottom))
        cv.rectangle(image, (int(left), int(top)), (int(right), int(bottom)), (23, 230, 210), thickness=2)
        left = left - 15 if left - 15 > 15 else left + 15
        top = top - 5
        cv.putText(image, label, (int(left), int(top)),
            cv.FONT_HERSHEY_SIMPLEX, 0.5,(0,255,0), 2)
            
plt.figure(figsize=(20,20))
plt.subplot(2,2,3)
plt.title("Detection")
plt.imshow(image)
# cv.imshow('image', image)
# cv.waitKey()


image = cv2.imread('crop.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
thresh = cv2.adaptiveThreshold(gray, 255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 2)
resizeImg = cv2.resize(image,None,fx=2,fy=2,interpolation=cv2.INTER_CUBIC)
finalImg = cv2.GaussianBlur(resizeImg,(5,5),0)
result = pytesseract.image_to_string(finalImg,lang ='eng',
config ='--oem 1 -l eng --psm 6')
print(result)

# plt.figure(figsize=(20,20))
plt.title(result)
plt.imshow(final)
