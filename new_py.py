import streamlit as st
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import cv2
import matplotlib.pyplot as plt
import time
from playsound import playsound
st.set_page_config(layout='wide', page_title='My Web App', page_icon=':smiley:', initial_sidebar_state='auto')
# labels = os.listdir('C:/Users/Ajay/Downloads/Drowsy Driver Data/Drowsy_Driver/train')
# import matplotlib.pyplot as plt
# def face_for_yawn(direc="C:/Users/Ajay/Downloads/Drowsy Driver Data/Drowsy_Driver/train", face_cas_path="C:/Users/Ajay/Downloads/Drowsy Driver Data/Drowsy_Driver/haarcascade_frontalcatface.xml"):
#     yaw_no = []
#     IMG_SIZE = 145
#     categories = ["yawn", "no_yawn"]
#     for category in categories:
#         path_link = os.path.join(direc, category)
#         class_num1 = categories.index(category)
#         print(class_num1)
#         for image in os.listdir(path_link):
#             image_array = cv2.imread(os.path.join(path_link, image), cv2.IMREAD_COLOR)
#             face_cascade = cv2.CascadeClassifier(face_cas_path)
#             faces = face_cascade.detectMultiScale(image_array, 1.3, 5)
#             for (x, y, w, h) in faces:
#                 img = cv2.rectangle(image_array, (x, y), (x+w, y+h), (0, 255, 0), 2)
#                 roi_color = img[y:y+h, x:x+w]
#                 resized_array = cv2.resize(roi_color, (IMG_SIZE, IMG_SIZE))
#                 yaw_no.append([resized_array, class_num1])
#     return yaw_no


# def get_data(dir_path="C:/Users/Ajay/Downloads/Drowsy Driver Data/Drowsy_Driver/train", face_cas="C:/Users/Ajay/Downloads/Drowsy Driver Data/Drowsy_Driver/haarcascade_frontalcatface.xml", eye_cas=""):
#     labels = ['Closed', 'Open']
#     IMG_SIZE = 145
#     data = []
#     for label in labels:
#         path = os.path.join(dir_path, label)
#         class_num = labels.index(label)
#         class_num +=2
#         print(class_num)
#         for img in os.listdir(path):
#             try:
#                 img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_COLOR)
#                 resized_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
#                 data.append([resized_array, class_num])
#             except Exception as e:
#                 print(e)
#     return data

# def append_data():
# #     total_data = []
#     yaw_no = face_for_yawn()
#     data = get_data()
#     yaw_no.extend(data)
#     return np.array(yaw_no)

# new_data = append_data()

# X = []
# y = []
# for feature, label in new_data:
#     X.append(feature)
#     y.append(label)

# X = np.array(X)
# X = X.reshape(-1, 145, 145, 3)

# from sklearn.preprocessing import LabelBinarizer
# label_bin = LabelBinarizer()
# y = label_bin.fit_transform(y)

# y = np.array(y)

# from sklearn.model_selection import train_test_split
# seed = 42
# test_size = 0.30
# X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=seed, test_size=test_size)

from tensorflow.keras.layers import Input, Lambda, Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
#from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf

import keras

# train_generator = ImageDataGenerator(rescale=1/255, zoom_range=0.2, horizontal_flip=True, rotation_range=30)
# test_generator = ImageDataGenerator(rescale=1/255)

# train_generator = train_generator.flow(np.array(X_train), y_train, shuffle=False)
# test_generator = test_generator.flow(np.array(X_test), y_test, shuffle=False)


model = Sequential()

# model.add(Conv2D(256, (3, 3), activation="relu", input_shape=X_train.shape[1:]))
# model.add(MaxPooling2D(2, 2))

# model.add(Conv2D(128, (3, 3), activation="relu"))
# model.add(MaxPooling2D(2, 2))

# model.add(Conv2D(64, (3, 3), activation="relu"))
# model.add(MaxPooling2D(2, 2))

# model.add(Conv2D(32, (3, 3), activation="relu"))
# model.add(MaxPooling2D(2, 2))

# model.add(Flatten())
# model.add(Dropout(0.5))

# model.add(Dense(64, activation="relu"))
# model.add(Dense(4, activation="softmax"))

# model.compile(loss="categorical_crossentropy", metrics=["accuracy"], optimizer="adam")

# model.summary()


# history = model.fit(train_generator, epochs=50, validation_data=test_generator, shuffle=True, validation_steps=len(test_generator))


# model.save("drowiness_new6.h5")

# model.save("drowiness_new6.model")

# prediction = model.predict_classes(X_test)

labels_new = ["yawn", "no_yawn", "Closed", "Open"]
IMG_SIZE = 145
def prepare(filepath, face_cas="C:/Users/nhegd/Desktop/JupyterNoteBooks/Drowsy Driver/haarcascade_frontalface_default.xml"):
    img_array = cv2.imread(filepath, cv2.IMREAD_COLOR)
    img_array = img_array / 255
    resized_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    return resized_array.reshape(-1, IMG_SIZE, IMG_SIZE, 3)

model = tf.keras.models.load_model("C:/Users/Ajay/Downloads/Drowsy Driver Data/drowiness_new6_1.h5")

from PIL import Image
# prepare("../input/drowsiness-dataset/train/no_yawn/1068.jpg")
prediction = model.predict([prepare("C:/Users/Ajay/Downloads/Drowsy Driver Data/Drowsy_Driver/train/Closed/_231.jpg")])
# temp=
beep_sound='C:/Users/Ajay/Downloads/Drowsy Driver Data/Drowsy_Driver/beep-01a.wav'
st.write(np.argmax(prediction))
if(np.argmax(prediction)==2):
    start_time=time.time()
    while(time.time()-start_time<6):
        playsound(beep_sound)
# if(temp==0):
#     st.write("Yawn")
# elif(temp==1):
#     st.write("No Yawn")
# elif(temp==2):
#     st.write("Closed Eyes")
# else:
#     st.write("Open Eyes")
uploaded_file="C:/Users/Ajay/Downloads/Drowsy Driver Data/Drowsy_Driver/train/Closed/_231.jpg"
image = Image.open(uploaded_file)
st.image(image, caption='Uploaded Image', use_column_width=True)
#st.image(plt.imshow(plt.imread("C:/Users/Ajay/Downloads/Drowsy Driver Data/Drowsy_Driver/train/no_yawn/4.jpg")))



