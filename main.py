import cv2
import mediapipe as mp
import numpy as np
from config import max_num_hands, alphabet_gesture
from gesture_recognition import load_gesture_model, recognize_gesture

# Initialize MediaPipe hands model
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=max_num_hands,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Load gesture recognition model
knn = load_gesture_model()

# Initialize variables
sentence = ""
last_character = ""
current_character = ""
gesture_count = 0
gesture_threshold = 3

# Start video capture
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, img = cap.read()
    if not ret:
        continue

    img = cv2.flip(img, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    result = hands.process(img)

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # Initialize hand_results
    hand_results = []

    if result.multi_hand_landmarks is not None:
        for res in result.multi_hand_landmarks:
            joint = np.zeros((21, 3))
            for j, lm in enumerate(res.landmark):
                joint[j] = [lm.x, lm.y, lm.z]

            gesture_result = recognize_gesture(knn, joint)
            if gesture_result:
                hand_results.append(gesture_result)

            mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)

        if len(hand_results) == 1:
            text = hand_results[0]
            temp = int(text)
            if temp in alphabet_gesture:
                text = alphabet_gesture[temp]
                if text != current_character:
                    gesture_count = 0  # Reset the gesture count if a different character is detected
                else:
                    gesture_count += 1  # Increment the gesture count if the same character is detected

                if gesture_count >= gesture_threshold:
                    if text != last_character:
                        sentence += text  # Add recognized letter to sentence
                        last_character = text
                    gesture_count = 0  # Reset the gesture count after adding the character
                current_character = text
            else:
                text = "?"

            (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, thickness=3)
            x, y = 10, 30
            cv2.rectangle(img, (x + 50, y - h + 50), (x + w + 50, y + 50), (100, 100, 100), -1)
            cv2.putText(img, text=text, org=(x + 50, y + 50), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, color=(255, 255, 255), thickness=3)

        elif len(hand_results) == 2:
            combined_text = hand_results[1] + hand_results[0]
            temp = int(combined_text)
            if temp in alphabet_gesture:
                text = alphabet_gesture[temp]
                if text != current_character:
                    gesture_count = 0  # Reset the gesture count if a different character is detected
                else:
                    gesture_count += 1  # Increment the gesture count if the same character is detected

                if gesture_count >= gesture_threshold:
                    if text != last_character:
                        sentence += text  # Add recognized letter to sentence
                        last_character = text
                    gesture_count = 0  # Reset the gesture count after adding the character
                current_character = text
            else:
                text = "?"

            (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, thickness=3)
            x, y = 10, 30
            cv2.rectangle(img, (x + 50, y - h + 50), (x + w + 50, y + 50), (100, 100, 100), -1)
            cv2.putText(img, text=text, org=(x + 50, y + 50), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, color=(255, 255, 255), thickness=3)

    # Display the current sentence
    cv2.putText(img, sentence, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3, cv2.LINE_AA)

    # Check for completion gesture (e.g., all fingers touching)
    if '0' in hand_results:
        # Display the completed sentence
        print("Completed Sentence: ", sentence)
        cv2.putText(img, "Completed: " + sentence, (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3, cv2.LINE_AA)
        # Clear the sentence
        sentence = ""
        last_character = ""
        current_character = ""
        gesture_count = 0

    cv2.imshow('Alphabet Recognition', img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
