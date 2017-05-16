# code to recognize an user in the ui
import cv2,os
import numpy as np
from PIL import Image

def RegisterUser():
    webcam = cv2.VideoCapture(0)
    detection=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    Number=input('enter your id')
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
            cv2.imwrite("C://Users/Austin/AppData/Local/Programs/Python/Python36-32/CST205-Project3-group254-austin-feature/Project 3 Facial Reco/pictures/"+Number +'.'+ str(exemple) + ".jpg", gray[y:y+h,x:x+w])

            cv2.imshow('frame',picture)
    #wait for 100 miliseconds each picture
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break
    # break at the 20 pictures
        elif exemple>30:
            break   
    webcam.release() 
    cv2.destroyAllWindows()

def LearnFace():
    #reload(LearnFace)
    reconaissance = cv2.face.createLBPHFaceRecognizer()
    detection = cv2.CascadeClassifier("haarcascade_frontalface_default.xml");

    def getImagesAndLabels(path): #path of all images in the folder
        imageplace=[os.path.join(path,f) for f in os.listdir(path)] 
        facetest=[]
        Num=[]

        for imageplace in imageplace:
            #converting picture in to gray scale
            pilImage=Image.open(imageplace).convert('L')
            #converting the PIL image into numpy array
            imageNp=np.array(pilImage,'uint8')
            #getting the Name from the image
            idname=int(os.path.split(imageplace)[-1].split(".")[1])
            # extract the face 
            visages=detection.detectMultiScale(imageNp)
            #If there is a face then append that in the list as well as Id of it
            for (x,y,w,h) in visages:
                facetest.append(imageNp[y:y+h,x:x+w])
                Num.append(idname)
        return facetest,Num

    visages,Num = getImagesAndLabels('pictures')
    reconaissance.train(visages, np.array(Num))
    reconaissance.save('trainner/trainner.yml')

def DetectFace():
    reconaissance = cv2.face.createLBPHFaceRecognizer()
    reconaissance.load('trainner/trainner.yml')
    chemin = "haarcascade_frontalface_default.xml"
    visagechemin = cv2.CascadeClassifier(chemin);


    webcam = cv2.VideoCapture(0)
    font = cv2.FONT_HERSHEY_SIMPLEX
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
            cv2.putText(im,str(Number), (x,y+h),font,2,(255,255,255),2,cv2.LINE_AA)
        cv2.imshow('im',im) 
        if cv2.waitKey(10) & 0xFF==ord('q'):
            break
    webcam.release()
    cv2.destroyAllWindows()
    
    
