import face_recognition
import cv2
import numpy as np
import csv
from datetime import datetime

#captures the vedio from webcam
video_capture= cv2.VideoCapture(0)

#Loading the known faces
number_1=face_recognition.load_image_file("faces/1.jpg")
encoded_number1=face_recognition.face_encodings(number_1)[0]

number_2=face_recognition.load_image_file("faces/2.jpg")
encoded_number2=face_recognition.face_encodings(number_2)[0]

known_face_encoadings=[encoded_number1,encoded_number2]
known_face_names=['Roll no 1','Roll no 2']

# list of expected students
students= known_face_names.copy()

face_locations=[]
face_encodings=[]

current=datetime.now()
current_date=current.strftime("%Y-%m-%d")

#write the present students in {current_date}.csv file
f= open(f"{current_date}.csv","a+",newline="")
writer=csv.writer(f)

while True:
    _, frame=video_capture.read()
    small_frame=cv2.resize(frame,(0,0),fx=0.25,fy=0.25)
    rgb_small_frames=cv2.cvtColor(small_frame,cv2.COLOR_BGR2RGB)

    # face recogonition
    face_locations=face_recognition.face_locations(rgb_small_frames)
    face_encodings=face_recognition.face_encodings(rgb_small_frames,face_locations)
    
    for face_encoding in face_encodings:
        #compare_faces() compares the encodings and return true and false 
        match=face_recognition.compare_faces(known_face_encoadings,face_encoding)
        #face_distance() find the distance between encodings
        face_distance=face_recognition.face_distance(known_face_encoadings,face_encoding)
        best_match_index=np.argmin(face_distance)#min distance

        if(match[best_match_index]):
            name=known_face_names[best_match_index]

        #for the text visible on the cam window
        if name in known_face_names:
            font=cv2.FONT_HERSHEY_SIMPLEX
            bottomleftCornerOfText=(10,100)
            fontScale=1.5
            fontColor=(255,0,0)
            thickness=3
            lineType=2
            cv2.putText(frame,name+"Present",bottomleftCornerOfText,font,fontScale,fontColor,thickness,lineType)

        #remove the name of present student from list
        if name in students:
            students.remove(name)
            current_time=current.strftime("%H-%M-%S")
            writer.writerow([name,current_time])
    # for exit of web cam video
    cv2.imshow("Attendace",frame)
    if cv2.waitKey(1) & 0xFF==ord("q"):
        break

video_capture.release()
cv2.distroyAllWindows()
f.close()
