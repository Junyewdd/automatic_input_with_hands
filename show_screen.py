import numpy as np
import cv2

def show_let_start_screen():
    # Create a black image
    img = np.zeros((480, 640, 3), np.uint8)
    # Write "Let's Start" text on the image
    (w, h), _ = cv2.getTextSize("Let's Start", cv2.FONT_HERSHEY_SIMPLEX, 2, 3)
    x, y = (img.shape[1] - w) // 2, (img.shape[0] + h) // 2
    cv2.putText(img, "Let's Start", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
    # Display the image
    cv2.imshow('Alphabet Recognition', img)
    cv2.waitKey(1000)  # Display the image for 3 seconds
    cv2.destroyAllWindows()
    
def show_hi_screen():
    # Create a black image
    img = np.zeros((480, 640, 3), np.uint8)
    # Write "Let's Start" text on the image
    (w, h), _ = cv2.getTextSize("Let's Start", cv2.FONT_HERSHEY_SIMPLEX, 2, 3)
    x, y = (img.shape[1] - w) // 2, (img.shape[0] + h) // 2
    cv2.putText(img, "HI", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
    # Display the image
    cv2.imshow('Alphabet Recognition', img)
    cv2.waitKey(3000)  # Display the image for 3 seconds
    cv2.destroyAllWindows()
    