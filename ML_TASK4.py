import cv2
import numpy as np

def preprocess_image(image):
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply adaptive thresholding to segment the hand from the background
    _, thresholded = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY_INV)
    
    # Find contours of the hand region
    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Find the largest contour (hand)
    hand_contour = max(contours, key=cv2.contourArea)
    
    # Create a bounding rectangle around the hand
    x, y, w, h = cv2.boundingRect(hand_contour)
    
    # Extract the hand region from the image
    hand_roi = thresholded[y:y+h, x:x+w]
    
    # Resize the hand region to a fixed size
    hand_roi = cv2.resize(hand_roi, (64, 64))
    
    # Normalize the pixel values to the range [0, 1]
    hand_roi = hand_roi / 255.0
    
    hand_roi = np.reshape(hand_roi, (1, 64, 64, 1))
    
    return hand_roi

def predict_gesture(model, image):
    hand_roi = preprocess_image(image)
    
    prediction = model.predict(hand_roi)
    
    gesture_label = np.argmax(prediction)
    
    return gesture_label
