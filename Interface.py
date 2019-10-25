import numpy
from pygame import mixer
import time
import cv2
from tkinter import *
import tkinter.messagebox
from tkinter import messagebox
import numpy as np
from joblib import load
from playsound import playsound
import pandas as pd
import matplotlib.pyplot as plt
from joblib import dump, load
from pathlib import Path

from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


root=Tk()
root.geometry('500x570')
frame = Frame(root, relief=RIDGE, borderwidth=2)
frame.pack(fill=BOTH,expand=1)
root.title('FaceRecognition')
frame.config(background='light blue')
label = Label(frame, text="FaceRecognition",bg='light blue',font=('Times 35 bold'))
label.pack(side=TOP)
filename = PhotoImage(file="D:\FaceReconition\demo.png")
background_label = Label(frame,image=filename)
background_label.pack(side=TOP)

def hel():
   help(cv2)

def Contri():
   tkinter.messagebox.showinfo("Contributors","\n1.Mayur Kadam\n2. Abhishek Ezhava \n3. Rajendra Patil \n")


def anotherWin():
   tkinter.messagebox.showinfo("About",'Driver Cam version v1.0\n Made Using\n-OpenCV\n-Numpy\n-Tkinter\n In Python 3')


def sa():
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
      if cv2.waitKey(1) & 0xFF == ord('q'):
         break
   capture.release()
   cv2.destroyAllWindows()
    

def fa():
   HAAR_MODEL = './model-haar/haarcascade_frontalface_default.xml'
   SVM_MODEL = './model-svm/faces.lib'

   font = cv2.FONT_HERSHEY_SIMPLEX
   color_deful = (255,100,0)
   color_unknown = (0,0,255)
   threshold = 0.8

   detector = cv2.CascadeClassifier(HAAR_MODEL)
   classifier = load(SVM_MODEL)

   capture = cv2.VideoCapture(0)
   while True:
      if capture is None:
         break
      ret, frame = capture.read()
      image = frame.copy()
      gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
      faces = detector.detectMultiScale(gray, 1.3, 5)         

      for (x, y, w, h) in faces:
         testset = []
         face = gray[y:y+h, x:x+w]
         face_resized = cv2.resize(face,(64,64),interpolation=cv2.INTER_LINEAR)
         testset.append(np.ravel(face_resized))

         pred = str(classifier.predict(testset))
         # compair matrx รูปภาพกับโมเดล
         prob = classifier.predict_proba(testset)        
         # predict matrix
        
         #print(prob[0]);
         #print(max(prob[0]));
         max_prob = max(prob[0])
        
         if max_prob >= threshold:
               text = ''.join(pred + ' (' + '{0:.2g}'.format(max_prob * 100) + '%)')
               color = color_deful
               if pred.find('deful') != -1:
                   color = color_deful        
               cv2.putText(image, text, (x,y-10), font, 0.6, color, thickness=2)
         else:
            color = color_unknown
            playsound('D:\FaceReconition\Sound\digital.mp3')
            
      cv2.rectangle(image, (x,y), (x+w,y+h), color, 2)
      cv2.imshow('face classifier', image)
      if cv2.waitKey(1) & 0xFF == ord('q'):
          break
   capture.release()
   cv2.destroyAllWindows()

menu = Menu(root)
root.config(menu=menu)

subm1 = Menu(menu)
menu.add_cascade(label="Tools",menu=subm1)
subm1.add_command(label="Open CV Docs",command=hel)

subm2 = Menu(menu)
menu.add_cascade(label="About",menu=subm2)
subm2.add_command(label="Driver Cam",command=anotherWin)
subm2.add_command(label="Contributors",command=Contri)


def ta():
   HAAR_MODEL = './model-haar/haarcascade_frontalface_default.xml'
   INPUT_IMAGE_PATH = './datasets/faces'
   OUTPUT_CSV_FILE = './datasets/faces.csv'
   PROCESSED_IMAGE_PATH = './datasets/processed'
   PROCESSED_CSV_FILE = './datasets/processed.csv'
   DETECTED_FACE_PATH = './datasets/cropped'
   DETECTED_CSV_FILE = './datasets/cropped.csv'
   OUTPUT_MODEL_NAME = './model-svm/faces.lib'

   def create_csv(dataset_path, output_csv):
      root_dir = Path(dataset_path)
      items = root_dir.iterdir()

      filenames = []
      labels = []
      print('reading image files ... ')
      for item in items:
         if item.is_dir():
            for file in item.iterdir():
                  if file.is_file():
                     print(str(file))
                     filenames.append(file)
                     labels.append(item.name)
      raw_data = {'filename': filenames, 'label': labels}
      df = pd.DataFrame(raw_data, columns=['filename','label'])
      df.to_csv(output_csv)
      print(len(filenames), 'image file(s) read')
      input("Press [ENTER] key to continue...")

   def resize(image, width=None, height=None):
      (h, w) = image.shape[:2]
      if width is None:
         r = height/float(h)
         dim = (int(w*r), height)
      else:
         r = width/float(w)
         dim = (width, int(h*r))
      return cv2.resize(image, dim, interpolation=cv2.INTER_LINEAR)

   def process_image(input_csv, output_csv, output_path_name):
      dataset = pd.read_csv(input_csv, sep=',')
      ids = dataset.values[:,0]
      names = dataset.values[:,1]
      labels = dataset.values[:,2]

      output_path = Path(output_path_name)
      if not output_path.exists():
         output_path.mkdir()

      filenames = []
      print('preprocessing images ... ')
      for item in names:
         input_path = Path(item)
         if input_path.is_file():
               output_name = output_path_name + '/image' + str(ids[len(filenames)]) + input_path.suffix
               print(input_path, '->', output_name)
               image = cv2.imread(str(input_path))
               image = resize(image, width=64)
               cv2.imwrite(output_name, image)
               filenames.append(output_name)
      prc_data = {'filename': filenames, 'label': labels}
      df = pd.DataFrame(prc_data, columns=['filename', 'label'])
      df.to_csv(output_csv)
      print(len(filenames), 'image file(s) processed')
      input("Press [ENTER] key to continue...")

   def detect_face(input_csv, output_csv, output_path_name):
      dataset = pd.read_csv(input_csv, sep=',')
      ids = dataset.values[:,0]
      names = dataset.values[:,1]
      labels = dataset.values[:,2]

      output_path = Path(output_path_name)
      if not output_path.exists():
         output_path.mkdir()

      clf = cv2.CascadeClassifier(HAAR_MODEL)
      face_filenames = []
      face_labels = []
      count = 0
      face_count = 0
      print('detecting faces ... ')
      for item in names:
         image = cv2.imread(item)
         face_label = labels[count]

         gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
         faces = clf.detectMultiScale(gray, 1.3, 5)
         for (x, y, w, h) in faces:
            cropped = image[y:y+h, x:x+w]
            output_file = output_path_name + '/face' + str(len(face_filenames)) + '.jpg'
            cv2.imwrite(output_file, cropped)

            face_filenames.append(output_file)
            face_labels.append(face_label)
         print(item, '->', len(faces), ' face(s) detected')
         face_count += len(faces)
         count += 1
      crp_data = {'filename': face_filenames, 'label': face_labels}
      df = pd.DataFrame(crp_data, columns=['filename', 'label'])
      df.to_csv(output_csv)
      print('Total of', face_count, 'face(s) detected')
      input("Press [ENTER] key to continue...")

   def train_model(train_csv, output_model_name):
      dataset = pd.read_csv(train_csv, sep=',')
      ids = dataset.values[:,0]
      names = dataset.values[:,1]
      labels = dataset.values[:,2]

      images = []
      print('Training recognition model ...')
      for item in names:
         image = cv2.imread(str(item), 0)
         resized = cv2.resize(image, (64,64), interpolation=cv2.INTER_LINEAR)
         images.append(np.ravel(resized))

      clf = SVC(kernel='linear', probability=True)
      clf.fit(images, labels)
      dump(clf, output_model_name)
      print('Model created in', output_model_name)
      input("Press [ENTER] key to continue...")

   def validate_model(validate_csv, model_name):
      dataset = pd.read_csv(validate_csv, sep=',')
      ids = dataset.values[:,0]
      names = dataset.values[:,1]
      labels = dataset.values[:,2]

      images = []
      print('Validating recognition model ...')
      for item in names:
         image = cv2.imread(str(item), 0)
         resized = cv2.resize(image, (64,64), interpolation=cv2.INTER_LINEAR)
         images.append(np.ravel(resized))

      clf = load(model_name)
      y_p = cross_val_predict(clf, images, labels, cv=3)
      print('Accuracy Score:', '{0:.4g}'.format(accuracy_score(labels,y_p) * 100), '%')
      print('Confusion Matrix:')
      print(confusion_matrix(labels,y_p))
      print('Classification Report:')
      print(classification_report(labels,y_p))

   create_csv(INPUT_IMAGE_PATH, OUTPUT_CSV_FILE)
   process_image(OUTPUT_CSV_FILE, PROCESSED_CSV_FILE, PROCESSED_IMAGE_PATH)
   train_model(PROCESSED_CSV_FILE, OUTPUT_MODEL_NAME)
   validate_model(PROCESSED_CSV_FILE, OUTPUT_MODEL_NAME)




def exitt():
   exit()


but1=Button(frame,padx=5,pady=5,width=39,bg='white',fg='black',relief=GROOVE,command=sa,text='Face-Auto-Save',font=('helvetica 15 bold'))
but1.place(x=5,y=104)

but2=Button(frame,padx=5,pady=5,width=39,bg='white',fg='black',relief=GROOVE,command=ta,text='Face-Trainer',font=('helvetica 15 bold'))
but2.place(x=5,y=176)

but3=Button(frame,padx=5,pady=5,width=39,bg='white',fg='black',relief=GROOVE,command=fa,text='Face-Recognition',font=('helvetica 15 bold'))
but3.place(x=5,y=250)

but5=Button(frame,padx=5,pady=5,width=5,bg='white',fg='black',relief=GROOVE,text='EXIT',command=exitt,font=('helvetica 15 bold'))
but5.place(x=210,y=478)


root.mainloop()