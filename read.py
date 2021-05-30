import cv2, os
import shutil
import csv
import numpy as np
from PIL import Image, ImageTk
import pandas as pd


name , Id = '',''
dic = {
    'Name' : name,
    'Ids' : Id
}
def store_data():
    global name,Id,dic
    name = str(input("Enter Name  "))
   
    Id  = str(input("Enter Id   "))
   
    dic = {
        'Ids' : Id,
        'Name': name
    }
    c = dic
    return  c

#Fucntion to check if entered ID is number or not
def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass
 
    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass
 
    return False

def TakeImages():
    dict1 = store_data()
    #name = "taru"
    #Id = '1'
    if (name.isalpha() and is_number(Id)):
        #Checking Id if it is 1 we are rewring the profile else just updating csv
        if Id == '1':
            fieldnames = ['Name','Ids']
            with open('Profile.csv','w') as f:
                writer = csv.DictWriter(f, fieldnames =fieldnames)
                writer.writeheader()
                writer.writerow(dict1)
        else:
            fieldnames = ['Name','Ids']
            with open('Profile.csv','a+') as f:
                writer = csv.DictWriter(f, fieldnames =fieldnames)
                #writer.writeheader()
                writer.writerow(dict1)
        

        #Haarcascade file for detctionof face
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        # capture frames from a camera
        cap = cv2.VideoCapture(0)
        sampleNum = 0
        # loop runs if capturing has been initialized.
        while 1:

            # reads frames from a camera
            ret, img = cap.read()

            # convert to gray scale of each frames
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Detects faces of different sizes in the input image
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                # To draw a rectangle in a face
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), 2)
                sampleNum = sampleNum + 1
                roi_gray = gray[y:y + h, x:x + w]
                roi_color = img[y:y + h, x:x + w]
                cv2.imwrite("E:\opencv\TrainingImage\ " + name + "." + Id + '.' + str(sampleNum) + ".jpg", gray[y:y + h, x:x + w])
                

                # Detects eyes of different sizes in the input image
                eyes = eye_cascade.detectMultiScale(roi_gray)
                sampleNum = sampleNum + 1

                # To draw a rectangle in eyes
                for (ex, ey, ew, eh) in eyes:
                    cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 127, 255), 2)

            # Display an image in a window
            cv2.imshow('Cpaturing Face for Login', img)

            if cv2.waitKey(100) & 0xFF == ord('q'):
                break
            # break if the sample number is morethan 60
            elif sampleNum > 60:
                break

        # Close the window
        cap.release()

        # De-allocate any associated memory usage
        cv2.destroyAllWindows()
        res = "Images Saved for Name : " + name + " with ID  " + Id
        print(res)
        print(' Images save location is E:\opencv\TrainingImage\ ')
      
        
    else:
        if(name.isalpha()):
            print('Enter Proper Id')
        elif(is_number(Id)):
            print('Enter Proper name')
        else:
            print('Enter Proper Id and Name')
                    
        

#TakeImages()

