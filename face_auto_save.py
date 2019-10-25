import cv2
import numpy as np
from pathlib import Path

OUTPUT_PATH = 'datasets/faces'
SIZE = (64,64)
MAX_CAPTURE = 20
detector = cv2.CascadeClassifier('model-haar/haarcascade_frontalface_default.xml')

font = cv2.FONT_HERSHEY_SIMPLEX
color = (0,255,0)

label = input('input name: ')
output_path = Path(OUTPUT_PATH)
if not output_path.exists():
    output_path.mkdir()
output_face_path = Path(OUTPUT_PATH + '/' + label)
if not output_face_path.exists():
    output_face_path.mkdir()    

count = 0
capture = cv2.VideoCapture(0)
cv2.waitKey(1)
while count < MAX_CAPTURE:
    ret, frame = capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, 1.2, 4)
    if len(faces) > 0:
        for (x, y, w, h) in faces:
            if w >= SIZE[0]:
                face = frame[y:y+h, x:x+w]
                output_name = OUTPUT_PATH +  '/' + label + '/img' + str(count) + '.jpg'
                face_cropped = frame[y:y+h, x:x+w]
                face_resized = cv2.resize(face_cropped, SIZE, interpolation=cv2.INTER_LINEAR)
                cv2.imwrite(output_name, face_resized)
                cv2.rectangle(frame, (x,y), (x+w,y+h), color, 2)
                cv2.putText(frame, ' all = ' + str(count) + ' / ' + str(MAX_CAPTURE), (x,y-10), font, 0.6, color, thickness=2)
                count += 1
    cv2.imshow('face-auto-save', frame)
    if cv2.waitKey(0) & 0xFF == ord('q'):
        break
capture.release()
cv2.destroyAllWindows()
