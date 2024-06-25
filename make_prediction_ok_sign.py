import cv2
import sqlite3
import numpy as np
from keras.models import load_model

# Load pre-trained face recognition model
faceRecognizer = cv2.face.LBPHFaceRecognizer_create()
faceRecognizer.read("models/trained_lbph_face_recognizer_model.yml")

# Load Haarcascade for face detection
faceCascade = cv2.CascadeClassifier("models/haarcascade_frontalface_default.xml")

# Load sign detection model
signModel = load_model("models/keras_Model.h5", compile=False)
class_names = open("models/labels.txt", "r").readlines()

# Constants for display
fontFace = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 0.6
fontColor = (255, 255, 255)
fontWeight = 2
fontBottomMargin = 30
nametagHeight = 50
faceRectangleBorderSize = 2
knownTagColor = (100, 180, 0)
unknownTagColor = (0, 0, 255)
knownFaceRectangleColor = knownTagColor
unknownFaceRectangleColor = unknownTagColor

# Initialize face recognition counters
recognition_count = {}
REQUIRED_RECOGNITION_COUNT = 5
face_recognized = False

# Open a connection to the first webcam
camera = cv2.VideoCapture(0)

# Start looping
while True:
    # Capture frame-by-frame
    ret, frame = camera.read()
    if not ret:
        break

    if not face_recognized:
        # Convert frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Detect faces
        faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # For each face found
        for (x, y, w, h) in faces:
            # Recognize the face
            customer_uid, confidence = faceRecognizer.predict(gray[y:y + h, x:x + w])
            customer_name = "Unknown"
            nametagColor = unknownTagColor
            faceRectangleColor = unknownFaceRectangleColor

            # If the face is recognized within the confidence range
            if 60 < confidence < 85:
                try:
                    conn = sqlite3.connect('customer_faces_data.db')
                    c = conn.cursor()
                    c.execute("SELECT customer_name FROM customers WHERE customer_uid = ?", (customer_uid,))
                    row = c.fetchone()
                except sqlite3.Error as e:
                    print("SQLite error:", e)
                    row = None
                finally:
                    if conn:
                        conn.close()

                if row:
                    customer_name = row[0].split(" ")[0]
                    nametagColor = knownTagColor
                    faceRectangleColor = knownFaceRectangleColor

                    # Update recognition count
                    if customer_uid not in recognition_count:
                        recognition_count[customer_uid] = 0
                    recognition_count[customer_uid] += 1

                    # Check if the face has been recognized enough times
                    if recognition_count[customer_uid] >= REQUIRED_RECOGNITION_COUNT:
                        face_recognized = True
                        current_customer_uid = customer_uid
                        print(f"Face recognized: {customer_name}")
                        break

            # Create rectangle around the face
            cv2.rectangle(frame, (x - 20, y - 20), (x + w + 20, y + h + 20), faceRectangleColor, faceRectangleBorderSize)
            # Display name tag
            cv2.rectangle(frame, (x - 22, y - nametagHeight), (x + w + 22, y - 22), nametagColor, -1)
            cv2.putText(frame, f"{customer_name}", (x, y - fontBottomMargin), fontFace, fontScale, fontColor, fontWeight)

    if face_recognized:
        # Sign detection logic
        ret, sign_image = camera.read()
        if not ret:
            break

        # Resize and normalize the sign image
        sign_image_resized = cv2.resize(sign_image, (224, 224), interpolation=cv2.INTER_AREA)
        sign_image_array = np.asarray(sign_image_resized, dtype=np.float32).reshape(1, 224, 224, 3)
        sign_image_normalized = (sign_image_array / 127.5) - 1

        # Predict the sign
        prediction = signModel.predict(sign_image_normalized)
        index = np.argmax(prediction)
        class_name = class_names[index].strip()
        confidence_score = prediction[0][index]

        # Display the sign prediction and confidence score
        cv2.putText(sign_image, f"Sign: {class_name[2:]} ({confidence_score*100:.2f}%)", (10, 30), fontFace, 1, (0, 255, 0), 2)
        cv2.imshow("Sign Detection", sign_image)

        # Check if the sign is "OK" and the confidence is above threshold
        if class_name[2:].lower() == "Ok_sign" and confidence_score > 0.75:  # Adjust confidence threshold as needed
            try:
                conn = sqlite3.connect('customer_faces_data.db')
                c = conn.cursor()
                c.execute("UPDATE customers SET confirm = 1 WHERE customer_uid = ?", (current_customer_uid,))
                conn.commit()
                print(f"{customer_name} confirmed")
            except sqlite3.Error as e:
                print("SQLite error:", e)
            finally:
                if conn:
                    conn.close()

            recognition_count = {}  # Reset recognition count
            face_recognized = False  # Resume face detection

    # Display the resulting frame
    if not face_recognized:
        cv2.imshow('Detecting Faces...', frame)

    # Exit loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all OpenCV windows
camera.release()
cv2.destroyAllWindows()