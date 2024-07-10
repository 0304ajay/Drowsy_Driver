import cv2
import tensorflow as tf
import numpy as np
import argparse
import time
from playsound import playsound
# Constants
IMG_SIZE = 145

# Load the pre-trained Keras model for drowsiness detection
model = tf.keras.models.load_model("C:/Users/Ajay/Downloads/Drowsy Driver Data/drowiness_new6_1.h5")

# Function to preprocess frames for drowsiness prediction
def prepare_frame(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    frame_resized = cv2.resize(frame_rgb, (IMG_SIZE, IMG_SIZE))
    resized_array = frame_resized / 255.0  # Normalize
    return np.expand_dims(resized_array, axis=0)  # Add batch dimension

# Function to detect faces and eyes
def detectAndDisplay(frame, face_cascade, eyes_cascade):
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.equalizeHist(frame_gray)

    # Detect faces
    faces = face_cascade.detectMultiScale(frame_gray)

    for (x, y, w, h) in faces:
        center = (x + w // 2, y + h // 2)
        frame = cv2.ellipse(frame, center, (w // 2, h // 2), 0, 0, 360, (255, 0, 255), 4)
        faceROI = frame_gray[y:y + h, x:x + w]

        # In each face, detect eyes
        eyes = eyes_cascade.detectMultiScale(faceROI)
        for (x2, y2, w2, h2) in eyes:
            eye_center = (x + x2 + w2 // 2, y + y2 + h2 // 2)
            radius = int(round((w2 + h2) * 0.25))
            frame = cv2.circle(frame, eye_center, radius, (255, 0, 0), 4)

        # Prepare the face ROI for drowsiness prediction
        face_roi = prepare_frame(frame[y:y+h, x:x+w])

        # Predict drowsiness
        prediction = model.predict(face_roi)
        # beep_sound = 'C:/Users/nhegd/Desktop/vscode/Python/Drowsy Driver/beep-01a.wav'
        if prediction[0][0] > 0.5:
            beep_sound = 'C:/Users/Ajay/Downloads/Drowsy Driver Data/Drowsy_Driver/beep-01a.wav'
            cv2.putText(frame, "Drowsy", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
            # start_time=time.time()
            # while(time.time()-start_time<1):
            #     playsound(beep_sound)
            # print('Wake Up!!!!')
        else:
            cv2.putText(frame, "Awake", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow('Capture - Face detection', frame)

# Main function for camera input
def main():
    # Argument parsing
    parser = argparse.ArgumentParser(description='Real-time drowsiness detection with face and eye detection.')
    parser.add_argument('--face_cascade', help='Path to face cascade XML.', default='C:/Users/Ajay/Downloads/Drowsy Driver Data/Drowsy_Driver/haarcascade_frontalcatface.xml')
    parser.add_argument('--eyes_cascade', help='Path to eyes cascade XML.', default='C:/Users/Ajay/Downloads/Drowsy Driver Data/Drowsy_Driver/haarcascade_eye_tree_eyeglasses.xml')
    parser.add_argument('--camera', help='Camera device number.', type=int, default=0)
    args = parser.parse_args()

    # Load cascades
    face_cascade = cv2.CascadeClassifier(cv2.samples.findFile(args.face_cascade))
    eyes_cascade = cv2.CascadeClassifier(cv2.samples.findFile(args.eyes_cascade))

    # Open camera
    camera_device = args.camera
    cap = cv2.VideoCapture(camera_device)

    if not cap.isOpened():
        print('--(!)Error opening video capture')
        return

    while True:
        ret, frame = cap.read()
        if frame is None:
            print('--(!) No captured frame -- Break!')
            break

        detectAndDisplay(frame, face_cascade, eyes_cascade)

        if cv2.waitKey(10) == 27:  # ESC key to exit
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
