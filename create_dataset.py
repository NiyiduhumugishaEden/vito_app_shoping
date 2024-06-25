import cv2
import time
import sqlite3
import os

def generate_uid():
    # Get current Unix timestamp in milliseconds
    current_millis = int(time.time() * 1000)
    # Convert to seconds by dividing by 1000
    current_seconds = current_millis // 1000
    return current_seconds

# Initialize face cascade
face_cascade = cv2.CascadeClassifier('models/haarcascade_frontalface_default.xml')
time.sleep(1)

# Connect to SQLite database
try:
    conn = sqlite3.connect('customer_faces_data.db')
    c = conn.cursor()
    #print("Successfully connected to the database")
except sqlite3.Error as e:
    print("SQLite error:", e)

# Create a table to store face data if it doesn't exist
try:
    c.execute('''CREATE TABLE IF NOT EXISTS customers
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, customer_uid TEXT, customer_name TEXT,confirm DEFAULT 0)''')
    #print("Table 'customers' created successfully")
except sqlite3.Error as e:
    print("SQLite error:", e)

# For each person, one face id
customer_name = input('Enter the Customer Name: ')
customer_uid = generate_uid()

# Wait for 2 seconds to be able to switch to the Webcam Window.
print("Please get your face ready!")
time.sleep(1)

# Initialize the camera
camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

# Initialize sample face image
start_time = time.time()
interval = 500  # Capture an image every 500 milliseconds
current_time = start_time
image_count = 0  # Total number of images captured

# Start looping
while True:
    ret, image = camera.read()
    if not ret:
        print("Failed to capture frame from the camera")
        break

    # Convert frame to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)

    # If faces are detected, break out of the loop
    if len(faces) > 0:
        break

    # Display the video frame with message
    cv2.putText(image, "No face detected. Please position yourself in front of the camera.", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.imshow("Waiting for Face Detection...", image)

    # Wait for a key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("User quit the program.")
        break

# Once a face is detected, proceed to capture images and insert into the database
if len(faces) > 0:
    print("Face detected. Proceeding to capture images.")

    while True:
        # Capture frame-by-frame
        ret, image = camera.read()
        if not ret:
            print("Failed to capture frame from the camera")
            break

        # Convert frame to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)

        # Loop through faces
        for (x, y, w, h) in faces:
            # Draw rectangle around the face
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # Display progress
            fontFace = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 0.6
            fontColor = (0, 255, 0)
            fontWeight = 2
            fontBottomMargin = 5

            cv2.putText(image, f"Generating image {image_count+1}", (x, y - fontBottomMargin), fontFace, fontScale, fontColor, fontWeight)

            # Check if enough time has passed to capture an image and if image count is less than 100
            if (time.time() - current_time) * 1000 >= interval and image_count < 201:
                # Generate filename without spaces
                image_name = f"data.{customer_uid}_{image_count+1}.jpg"
                image_path = os.path.join('dataset', image_name)
                
                # Save the captured image
                cv2.imwrite(image_path, gray[y:y + h, x:x + w])
                current_time = time.time()
                image_count += 1

                # Save face data to database


        # Display the video frame with rectangle
        cv2.imshow("Dataset Generating...", image)

        # To stop taking video, press 'q' key or if image count reaches 100
        if cv2.waitKey(1) & 0xFF == ord('q') or image_count >= 50:
            try:
                c.execute("INSERT INTO customers (customer_uid, customer_name) VALUES (?, ?)", (customer_uid, customer_name))
                conn.commit()
                #print("Image inserted into database successfully")
            except sqlite3.Error as e:
                print("SQLite error:", e)
            break

# Release the camera and close all windows
camera.release()
cv2.destroyAllWindows()

# Close the database connection
conn.close()