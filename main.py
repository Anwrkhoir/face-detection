import cv2 as cv
from cvzone.ClassificationModule import Classifier

# Load the model and labels
model_path = "keras_model.h5" 
labels_path = "labels.txt"  
data = Classifier(model_path, labels_path)

# Load labels from file
with open(labels_path, 'r') as f:
    labels = f.read().strip().split('\n')

# Load Haar cascade for face detection
face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Video capture setup
cap = cv.VideoCapture(0)
cap.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 720)

while True:
    ret, img = cap.read()
    
    if ret:
        # Convert the image to grayscale for face detection
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        # Detect faces in the image
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            # Extract the region of interest (the face)
            face_img = img[y:y+h, x:x+w]

            # Get predictions for the face regionpy
            predict, index = data.getPrediction(face_img)

            # Confidence scores for each class
            confidence_scores = predict

            # Sum of all confidence scores
            total_confidence = sum(confidence_scores)

            # Calculate percentage detection for each class
            percentage_detections = [(score / total_confidence) * 100 for score in confidence_scores]

            max_index = percentage_detections.index(max(percentage_detections))
            max_percentage = percentage_detections[max_index]
            class_name = labels[max_index]

            print(f"Class {class_name}: {max_percentage:.2f}%")
            text = f"Class {class_name}: {max_percentage:.2f}%"
            position = (x, y - 10)
            cv.putText(img, text, position, cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv.LINE_AA)

            # Draw the bounding box around the face
            img = cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Show the image with bounding box
        cv.imshow("Kamera", img)

        # Wait for the user to press the ESC key to exit
        key = cv.waitKey(1)
        if key == 27:
            break
    else:
        print("Invalid frame captured")
        break

# Release resources
cap.release()
cv.destroyAllWindows()