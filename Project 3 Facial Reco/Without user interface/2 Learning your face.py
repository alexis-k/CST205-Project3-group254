# Learning, take the 30 pictures and creat the average Face. More you take picture more the precision is better
import cv2,os
import numpy as np
from PIL import Image

reconaissance = cv2.createLBPHFaceRecognizer()
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
# save in database
