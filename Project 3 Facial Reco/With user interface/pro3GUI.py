# this is the code from all user interface.
from tkinter import *
from tkinter import messagebox
import cv2,os
import numpy as np
from PIL import Image
from facefunctions import RegisterUser, LearnFace, DetectFace

def SuccessWindow():
    msg = messagebox.showinfo("Success!")

def secondWindow():
    sw = Tk()
    sw.geometry("200x200")
    a = Button(sw, text = "1. Open Camera (hold still!)", command = RegisterUser)
    a.pack()
    b = Button(sw, text = "2. Complete Registration", command = LearnFace)
    b.pack()
    c = Button(sw, text = "Click when done", command = sw.destroy)
    c.pack()

    sw.mainloop()






top = Tk()
top.geometry("150x150")
w = Button (top, text = "Create New User", command = secondWindow)
w.pack()
y = Button (top, text = "Existing User", command = DetectFace)
y.pack()


top.mainloop()

