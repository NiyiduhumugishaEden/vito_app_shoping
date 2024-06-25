# Virtual Try-On (ViTO) Project 👌👌

---

## Overview

The **Virtual Try-On (ViTO)** project enables a virtual shopping experience where customers interact with products using facial recognition and gesture detection. Customers are recognized based on their faces captured via webcam, and they can interact with virtual products by displaying gestures like an "OK" sign. This README provides an overview of the project, setup instructions, usage guidelines, and integration with Arduino for Bluetooth communication. 😎😎

---

## Files and Structure

The project includes several key files and directories:

- **01_create_dataset.py**: Captures grayscale images of customers' faces using a webcam, saves them, and stores metadata (customer name, image path) in a SQLite database.
  
- **02_create_clusters.py**: Automates the organization of face images into clusters based on features extracted from a VGG16 model, ensuring dataset cleanliness.
  
- **03_rearrange_data.py**: Cleans data by moving images to the 'dataset' directory, checking data integrity against the SQLite database, and maintaining consistency.
  
- **04_train_model.py**: Trains the LBPH face recognizer using dataset images, detects faces with the Haar Cascade Classifier, and saves the trained model.
  
- **05_make_predictions.py**: Implements real-time face recognition using the trained LBPH model, detects faces via webcam feed, predicts customer identities, and manages interactions with virtual products.
  
- **shop.py**: Integrates face recognition with product interaction, allowing recognized customers to interact with virtual products by displaying gestures like the "OK" sign.
  
- **models/**: Directory with trained models (LBPH face recognizer, product detection).

- **Glasses/**: Directory with overlay images (sunglasses) for virtual try-on.

- **customer_faces_data.db**: SQLite database storing customer data (names, IDs, recognition metadata) and shopping cart information.

- **arduino_communication.ino**: Arduino script enabling Bluetooth communication with the ViTO application using an HR-05 Bluetooth module.

- **dataset/**: Directory to be created for storing captured face images.

---

## Dependencies

Ensure these dependencies are installed to run the project:

- **Python 3.x**
- **OpenCV** (`opencv-python`)
- **NumPy** (`numpy`)
- **Keras** (`keras`)
- **SQLite3** (`sqlite3`)
- **cvzone** (for image overlay)
- **Arduino IDE** for uploading `arduino_communication.ino` to the Arduino board

Install dependencies using pip:

- **pip install opencv-python numpy keras sqlite3 cvzone
Setup Instructions
1. Clone the repository
bash
Copy code
git clone <repository-url>
cd vito-project
2. Create the dataset directory
Before running 01_create_dataset.py, create a directory named dataset in the root of the project:

bash
Copy code
mkdir dataset
3. Setup the dataset
Run 01_create_dataset.py to capture customer faces, save them into the dataset directory, and populate the SQLite database with customer metadata.

4. Train the face recognition model
Execute 04_train_model.py to train the LBPH face recognizer using the captured dataset.

5. Upload Arduino script
Upload arduino_communication.ino to the Arduino board connected to the HR-05 Bluetooth module.

6. Run the application
Launch shop.py to start the Virtual Try-On (ViTO) application. Ensure the webcam is connected and functioning properly.

7. Interact with the application
Position yourself in front of the webcam.
The system recognizes your face and displays your dynamically fetched name from the SQLite database.
Interact with virtual products by displaying gestures (e.g., "OK" sign) to add them to the shopping cart.
Use the 'q' key to exit the application.
Arduino Integration
Connect Arduino
Connect the Arduino board with the HR-05 Bluetooth module to the computer running the ViTO application.

Upload Script
Use the Arduino IDE to open arduino_communication.ino, modify if necessary, and upload it to the Arduino board.

Communication
Arduino receives Bluetooth messages from the ViTO application. Ensure Bluetooth communication settings (e.g., baud rate) match in both the Arduino script and Python code.

Contributing
Contributions are welcome! Fork the repository, create a pull request with proposed changes.

License
This project is licensed under the MIT License.

IF YOU LIKE IT LEAVE A STAR 🤣👍

Made with Love ❤️ by Niyiduhumugisha Eden
