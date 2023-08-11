import cv2
import cvzone
from cvzone.SelfiSegmentationModule import SelfiSegmentation
import os

cap=cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)
cap.set(cv2.CAP_PROP_FPS, 60)
segmentor = SelfiSegmentation()
fpsReader =cvzone.FPS()
imgBg = cv2.imread("backgrounds/p.png")
while True:
    success, img = cap.read()
    imgOut =segmentor.removeBG(img,(imgBg ),threshold=0.5)         
    imgStacked=cvzone.stackImages([img,imgOut],2,1)
    _,imgStacked =fpsReader.update(imgStacked,color=(0,0,255))   
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break 
    cv2.imshow("Image", imgStacked)
    cv2.waitKey(1)