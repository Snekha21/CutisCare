import cv2
import tensorflow as tf
import numpy as np

# Load the trained model
model = tf.keras.models.load_model('orginal_skin_model.h5')

# Define the labels for the classes
class_names = ['Acne and Rosacea Photos', 'vitiligo', 'Normal', 'Tinea Ringworm Candidiasis and other Fungal Infections', 'Melanoma Skin Cancer Nevi and Mole', 'Eczema']

# Load the VGG16 model without top layers
vgg_model = tf.keras.applications.VGG16(include_top=False, weights='imagenet', input_shape=(224, 224, 3))

# Define a function to preprocess the input image
def preprocess_image(img):
    # Resize the image to match the input size of the VGG model
    img = cv2.resize(img, (224, 224))
    # Convert the image to a numpy array and scale the pixel values to be between 0 and 1
    img = np.array(img) / 255.0
    return img

# Define a function to perform classification on a single frame
def classify_frame(frame):
    # Preprocess the frame
    img = preprocess_image(frame)
    # Perform feature extraction using VGG model
    features = vgg_model.predict(np.expand_dims(img, axis=0))
    # Flatten the features
    flattened_features = features.flatten()
    # Perform classification on the flattened features
    predictions = model.predict(np.expand_dims(flattened_features, axis=0))
    # Get the index of the class with the highest probability
    class_index = np.argmax(predictions)
    # Get the corresponding class label
    class_label = class_names[class_index]
    return class_label

# Define a function to capture frames from the laptop camera and perform classification in real-time
def live_classification():
    # Open the laptop camera
    cap = cv2.VideoCapture(0)
    while True:
        # Capture a frame from the camera
        ret, frame = cap.read()
        if not ret:
            break
        # Flip the frame horizontally to match the mirror-like view from the camera
        frame = cv2.flip(frame, 1)
        # Perform classification on the frame
        class_label = classify_frame(frame)
        # Draw the class label on the frame
        cv2.putText(frame, class_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        # Show the frame
        cv2.imshow('Live Classification', frame)
        # Exit the loop if the 'q' key is pressed
        if cv2.waitKey(1) == ord('q'):
            break
    # Release the camera and close all windows
    cap.release()
    cv2.destroyAllWindows()

# Start the live classification
live_classification()
