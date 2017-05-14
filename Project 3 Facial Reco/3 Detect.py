# Lat part, detect if it's the good user who is register. 
import cv2
import numpy as np

reconaissance = cv2.createLBPHFaceRecognizer()
reconaissance.load('trainner/trainner.yml')
chemin = "haarcascade_frontalface_default.xml"
visagechemin = cv2.CascadeClassifier(chemin);


webcam = cv2.VideoCapture(0)
font = cv2.cv.InitFont(cv2.cv.CV_FONT_HERSHEY_SIMPLEX, 1, 1, 0, 1, 1)
while True:
    ret, im =webcam.read()
    gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    visage=visagechemin.detectMultiScale(gray, 1.2,5)
    for(x,y,w,h) in visage:
        cv2.rectangle(im,(x,y),(x+w,y+h),(225,255,0),4)
        Number, cf = reconaissance.predict(gray[y:y+h,x:x+w])
        if(cf<50):
            if(Number==1):
                Number="You !"
            
        else:
            Number="Not Recognize"
        cv2.cv.PutText(cv2.cv.fromarray(im),str(Number), (x,y+h),font, 255)
    cv2.imshow('im',im) 
    if cv2.waitKey(10) & 0xFF==ord('q'):
        break
webcam.release()
cv2.destroyAllWindows()
