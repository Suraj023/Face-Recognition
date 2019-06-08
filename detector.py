import cv2
import numpy as np

faceDetect=cv2.CascadeClassifier('haarcascade_frontalface_default.xml');
cam=cv2.VideoCapture(0)
rec=cv2.face.LBPHFaceRecognizer_create();
rec.read("recognizer\\trainningData.yml")
id=0
sample=0
while(True):
    ret,img=cam.read()
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=faceDetect.detectMultiScale(gray,1.3,5);
    for(x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        id,conf=rec.predict(gray[y:y+h,x:x+h])
        if id==1:
            id="suraj"
        else:
            id="Don't know"
        #cv2.putText(img,str(id),(x,y+h),fontsca,255)
        cv2.putText(img,str(id),(x,y+h),cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255))

  
    cv2.imshow('Face',img)
    if cv2.waitKey(1)==ord('q'):
          break;
cam.release()
cv2.destroyAllWindows()
        
