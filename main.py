import cv2
import numpy as np
from show_screen import show_let_start_screen, show_hi_screen
from gesture_use import recognize_gestures, print_completed_sentences, use_gestures

if __name__ == "__main__":
    # show_hi_screen()
    # Step 1: Recognize gestures and form sentences
    recognize_gestures()

    # Step 2: Show the "Let's Start" screen
    show_let_start_screen()

    # Step 3: Recognize gestures and display sentences from completed_sentences
    use_gestures()

    # Step 4: Print all completed sentences
    print_completed_sentences()
