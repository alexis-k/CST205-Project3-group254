# Open your cam and detect the face of the user. Take 30 pictures every 100 millisecond and stock in file.
# to be use later for the learning.
import cv2
webcam = cv2.VideoCapture(0)
detection=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

Number=raw_input('enter your id')
exemple=0
while(True):
    ret, picture = webcam.read()
    gray = cv2.cvtColor(picture, cv2.COLOR_BGR2GRAY)
    faces = detection.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(picture,(x,y),(x+w,y+h),(255,255,0),4)
        
#incrementing exemple number 
        exemple=exemple+1
#saving  face in  folder
        cv2.imwrite("/home/alexis/Bureau/project/pictures/picture."+Number +'.'+ str(exemple) + ".jpg", gray[y:y+h,x:x+w])

        cv2.imshow('frame',picture)
#wait for 100 miliseconds each picture
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break
# break at the 20 pictures
    elif exemple>30:
        break
webcam.release()
cv2.destroyAllWindows()
