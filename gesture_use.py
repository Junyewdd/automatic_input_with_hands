import cv2
import mediapipe as mp
import numpy as np
from gesture_recognition import load_gesture_model, recognize_gesture
from config import max_num_hands, alphabet_gesture

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
completed_sentences = []  # Array to store completed sentences

def recognize_gestures(add_to_sentences=True):
    global sentence, last_character, current_character, gesture_count

    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, img = cap.read()
        if not ret:
            continue

        img = cv2.flip(img, 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        result = hands.process(img)

        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # Expand the image to the right to display the completed sentences
        expanded_img = np.zeros((img.shape[0], img.shape[1] + 300, 3), dtype=np.uint8)
        expanded_img[:, :img.shape[1]] = img

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

                # Draw landmarks on the original image
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

        # Copy the processed original image into the expanded image
        expanded_img[:, :img.shape[1]] = img

        # Display the current sentence
        cv2.putText(expanded_img, sentence, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3, cv2.LINE_AA)

        # Display the completed sentences on the right side of the screen
        for i, completed_sentence in enumerate(completed_sentences):
            cv2.putText(expanded_img, str(i) + " : " + completed_sentence, (img.shape[1] + 10, 30 * (i + 1)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Check for completion gesture (e.g., all fingers touching)
        if '0' in hand_results and sentence != '' and sentence not in completed_sentences:
            if add_to_sentences:
                # Add the completed sentence to the array
                completed_sentences.append(sentence)
                print("Completed Sentence: ", sentence)
            # else:
            #     if completed_sentences:
            #         sentence = completed_sentences.pop(0)
            #         print("Displaying Sentence: ", sentence)

            # Clear the sentence
            sentence = ""
            last_character = ""
            current_character = ""
            gesture_count = 0

        cv2.imshow('Alphabet Recognition', expanded_img)
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    
    return completed_sentences

def use_gestures():
    global sentence, last_character, current_character, gesture_count

    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, img = cap.read()
        if not ret:
            continue

        img = cv2.flip(img, 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        result = hands.process(img)

        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # Expand the image to the right to display the completed sentences
        expanded_img = np.zeros((img.shape[0], img.shape[1] + 300, 3), dtype=np.uint8)
        expanded_img[:, :img.shape[1]] = img

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

                # Draw landmarks on the original image
                mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)

            if len(hand_results) == 1:
                text = hand_results[0]
                temp = int(text)
                if temp < len(completed_sentences):
                    text = completed_sentences[temp]
                else:
                    text = "?"

                (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, thickness=3)
                x, y = 10, 30
                cv2.rectangle(img, (x + 50, y - h + 50), (x + w + 50, y + 50), (100, 100, 100), -1)
                cv2.putText(img, text=text, org=(x + 50, y + 50), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, color=(255, 255, 255), thickness=3)

            elif len(hand_results) == 2:
                combined_text = hand_results[1] + hand_results[0]
                temp = int(combined_text)
                if temp < len(completed_sentences):
                    text = completed_sentences[temp]
                else:
                    text = "?"

                (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, thickness=3)
                x, y = 10, 30
                cv2.rectangle(img, (x + 50, y - h + 50), (x + w + 50, y + 50), (100, 100, 100), -1)
                cv2.putText(img, text=text, org=(x + 50, y + 50), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, color=(255, 255, 255), thickness=3)

        # Copy the processed original image into the expanded image
        expanded_img[:, :img.shape[1]] = img

        # Display the current sentence
        cv2.putText(expanded_img, sentence, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3, cv2.LINE_AA)

        # Display the completed sentences on the right side of the screen
        for i, completed_sentence in enumerate(completed_sentences):
            cv2.putText(expanded_img, str(i) + " : " + completed_sentence, (img.shape[1] + 10, 30 * (i + 1)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # # Check for completion gesture (e.g., all fingers touching)
        # if '0' in hand_results and sentence != '':
        #     if add_to_sentences:
        #         # Add the completed sentence to the array
        #         completed_sentences.append(sentence)
        #         print("Completed Sentence: ", sentence)
            # else:
            #     if completed_sentences:
            #         sentence = completed_sentences.pop(0)
            #         print("Displaying Sentence: ", sentence)

            # Clear the sentence
            sentence = ""
            last_character = ""
            current_character = ""
            gesture_count = 0

        cv2.imshow('Alphabet Recognition', expanded_img)
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def print_completed_sentences():
    print("All completed sentences:")
    for s in completed_sentences:
        print(s)


# 잘못 들어간 문장 삭제
def delete_sentences():
            # Check for completion gesture (e.g., all fingers touching)
    if '0' in hand_results and sentence != '':
        if add_to_sentences:
            # Add the completed sentence to the array
            completed_sentences.append(sentence)
            print("Completed Sentence: ", sentence)
        else:
            if completed_sentences:
                sentence = completed_sentences.pop(0)
                print("Displaying Sentence: ", sentence)