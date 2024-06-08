import numpy as np
import cv2
import tkinter as tk
from tkinter import messagebox

def show_let_start_screen():
    # Create a black image
    img = np.zeros((480, 640, 3), np.uint8)
    # Write "Let's Start" text on the image
    (w, h), _ = cv2.getTextSize("Let's Start", cv2.FONT_HERSHEY_SIMPLEX, 2, 3)
    x, y = (img.shape[1] - w) // 2, (img.shape[0] + h) // 2
    cv2.putText(img, "Let's Start", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
    # Display the image
    cv2.imshow('Alphabet Recognition', img)
    cv2.waitKey(1000)  # Display the image for 1 second
    cv2.destroyAllWindows()

def show_hi_screen():
    root = tk.Tk()
    root.title("Shortcut Saver")
    root.geometry("300x200")

    label = tk.Label(root, text="Welcome! Choose an option to save shortcuts:")
    label.pack(pady=20)

    choice_var = tk.IntVar()  # Variable to store user's choice

    def option_1():
        messagebox.showinfo("Info", "You chose to recognize gestures.")
        choice_var.set(1)  # Set choice to 1
        root.destroy()  # Close the tkinter window

    def option_2():
        messagebox.showinfo("Info", "You chose to input shortcuts manually.")
        choice_var.set(2)  # Set choice to 2
        root.destroy()  # Close the tkinter window

    button1 = tk.Button(root, text="1. Recognize Gestures", command=option_1)
    button1.pack(pady=10)

    button2 = tk.Button(root, text="2. Input Shortcuts Manually", command=option_2)
    button2.pack(pady=10)

    root.mainloop()

    return choice_var.get()  # Return the user's choice

def input_shortcut():
    # Example function to input shortcuts manually
    print("Input shortcuts manually.")

def recognize_gestures():
    # Example function to recognize gestures
    print("Recognizing gestures.")
    

def show_thanks_screen():
    # Create a black image
    img = np.zeros((480, 640, 3), np.uint8)
    # Write "Let's Start" text on the image
    (w, h), _ = cv2.getTextSize("Let's Start", cv2.FONT_HERSHEY_SIMPLEX, 2, 3)
    x, y = (img.shape[1] - w) // 2, (img.shape[0] + h) // 2
    cv2.putText(img, "Thank you", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
    # Display the image
    cv2.imshow('Alphabet Recognition', img)
    cv2.waitKey(1000)  # Display the image for 1 second
    cv2.destroyAllWindows()