import cv2
import sqlite3
import numpy as np
from keras.models import load_model
import cvzone
import serial  # Import serial library for Bluetooth communication

# Load pre-trained face recognition model
faceRecognizer = cv2.face.LBPHFaceRecognizer_create()
faceRecognizer.read("models/trained_lbph_face_recognizer_model.yml")

# Load Haarcascade for face detection
faceCascade = cv2.CascadeClassifier("models/haarcascade_frontalface_default.xml")

# Load sign detection model
signModel = load_model("models/keras_Model.h5", compile=False)
class_names = open("models/labels.txt", "r").readlines()

# Load sunglasses overlays
overlays = [cv2.imread(f'Glasses/glass{i}.png', cv2.IMREAD_UNCHANGED) for i in range(1, 30)]

# Load Haarcascade for eye detection
eyeCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye_tree_eyeglasses.xml')

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
current_customer_uid = None

# Initialize OK sign detection counter
ok_sign_count = 0
REQUIRED_OK_SIGN_COUNT = 5

# Open a connection to the first webcam
camera = cv2.VideoCapture(0)

# Initialize variables for sunglasses fitting
num = 0
count = 0

# Function to create the cart table
def create_cart_table():
    try:
        conn = sqlite3.connect('customer_faces_data.db')
        c = conn.cursor()
        c.execute('''
            CREATE TABLE IF NOT EXISTS cart (
                customer_uid INTEGER,
                product_id INTEGER,
                quantity INTEGER,
                PRIMARY KEY (customer_uid, product_id)
            )
        ''')
        conn.commit()
    except sqlite3.Error as e:
        print("SQLite error:", e)
    finally:
        if conn:
            conn.close()

# Create the cart table
create_cart_table()

# Initialize Bluetooth serial communication
bluetooth_port = 'COM18'  # Replace with your Bluetooth COM port
baud_rate = 9600  # Standard baud rate for Bluetooth modules
bluetooth = None

try:
    bluetooth = serial.Serial(bluetooth_port, baud_rate)
    print(f"Successfully connected to Bluetooth on {bluetooth_port}")
except serial.SerialException as e:
    print(f"Failed to connect to Bluetooth on {bluetooth_port}: {e}")

# Main video capture and processing loop
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
                        # Send recognition success message over Bluetooth
                        if bluetooth:
                            bluetooth.write(b'Face Recognized Successfully\r\n')
                        break
            
            # Create rectangle around the face
            cv2.rectangle(frame, (x - 20, y - 20), (x + w + 20, y + h + 20), faceRectangleColor, faceRectangleBorderSize)
            
            # Display name tag
            cv2.rectangle(frame, (x - 22, y - nametagHeight), (x + w + 22, y - 22), nametagColor, -1)
            cv2.putText(frame, f"{customer_name}", (x, y - fontBottomMargin), fontFace, fontScale, fontColor, fontWeight)
        
        # Display the resulting frame
        cv2.imshow('Detecting Faces...', frame)
    
    if face_recognized:
        # Sunglasses fitting and sign detection logic
        ret, sign_image = camera.read()
        if not ret:
            break
        
        # Convert frame to grayscale
        gray_scale = cv2.cvtColor(sign_image, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = faceCascade.detectMultiScale(gray_scale, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        for (x, y, w, h) in faces:
            roi_gray = gray_scale[y:y + h, x:x + w]
            
            # Detect eyes within the face region
            eyes = eyeCascade.detectMultiScale(roi_gray, scaleFactor=1.3, minNeighbors=5)
            
            # Resize the current overlay to fit the detected face
            overlay_resize = cv2.resize(overlays[num], (w, int(h * 0.8)))
            
            # Overlay the resized sunglasses onto the frame
            sign_image = cvzone.overlayPNG(sign_image, overlay_resize, [x, y])
            
            # Blink detection logic
            if len(eyes) < 2:
                count += 1
                print(f"{count}: Blink Detected")
                if count == 5:
                    num = (num + 1) % 29  # Cycle through the overlays
                    count = 0
                    ok_sign_count = 0  # Reset OK sign count when sunglasses change
                cv2.waitKey(1000)  # Wait for 1 second
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
        
        # Check if the sign is "OK" and the confidence is above threshold
        if class_name[2:].lower() == "ok_sign" and confidence_score > 0.75:  # Adjust confidence threshold as needed
            ok_sign_count += 1
            print(f"OK sign detected: {ok_sign_count}")
            
            if ok_sign_count >= REQUIRED_OK_SIGN_COUNT:
                try:
                    conn = sqlite3.connect('customer_faces_data.db')
                    c = conn.cursor()
                    
                    # Check if the sunglasses are already in the cart
                    c.execute("SELECT quantity FROM cart WHERE customer_uid = ? AND product_id = ?", (current_customer_uid, num))
                    row = c.fetchone()
                    
                    if row:
                        # Update the quantity if the sunglasses are already in the cart
                        print("Found the sunglasses in cart")
                        new_quantity = row[0] + 1
                        c.execute("UPDATE cart SET quantity = ? WHERE customer_uid = ? AND product_id = ?", (new_quantity, current_customer_uid, num))
                    else:
                        # Insert a new record if the sunglasses are not in the cart
                        print("Inserting the sunglasses in cart")
                        c.execute("INSERT INTO cart (customer_uid, product_id, quantity) VALUES (?, ?, ?)", (current_customer_uid, num, 1))
                        c.execute("UPDATE customers SET confirm = 1 WHERE customer_uid = ?", (current_customer_uid,))
                        conn.commit()
                        print(f"{customer_name} confirmed and sunglasses updated in cart")
                
                except sqlite3.Error as e:
                    print("SQLite error:", e)
                
                finally:
                    if conn:
                        conn.close()
                
                # Send message via Bluetooth on successful shopping
                if bluetooth:
                    bluetooth.write(b'Successful Shopping\r\n')  # Adjust message as needed
                
                recognition_count = {}  # Reset recognition count
                face_recognized = False  # Resume face detection
                ok_sign_count = 0  # Reset OK sign count
        
        else:
            ok_sign_count = 0  # Reset OK sign count if not detected
        
        # Display the OK sign counter
        cv2.putText(sign_image, f"OK Sign Counter: {ok_sign_count}/{REQUIRED_OK_SIGN_COUNT}", (10, 60), fontFace, 1, (255, 0, 0), 2)
        
        # Display the frame
        cv2.imshow('Sunglasses Fitting and Sign Detection', sign_image)
    
    # Exit loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all OpenCV windows
camera.release()
cv2.destroyAllWindows()
