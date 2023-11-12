import cv2
import mediapipe as mp
import math
import pyttsx3

# Initialize Mediapipe hands module
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
engine = pyttsx3.init()


# Initialize the camera
cap = cv2.VideoCapture(0)

with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            continue
        
        # Convert the frame to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame to detect hands
        results = hands.process(rgb_frame)
        
        if results.multi_hand_landmarks:
            for landmarks in results.multi_hand_landmarks:
                # Draw landmarks on the frame
                mp_drawing.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)
                
                # Get the coordinates of specific landmarks
                thumb_tip = landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                index_finger_tip = landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                middle_finger_tip = landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
                ring_finger_tip = landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
                pinky_tip = landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
                
                # Calculate distances between finger tips
                thumb_to_index = math.dist((thumb_tip.x, thumb_tip.y), (index_finger_tip.x, index_finger_tip.y))
                index_to_middle = math.dist((index_finger_tip.x, index_finger_tip.y), (middle_finger_tip.x, middle_finger_tip.y))
                middle_to_ring = math.dist((middle_finger_tip.x, middle_finger_tip.y), (ring_finger_tip.x, ring_finger_tip.y))
                ring_to_pinky = math.dist((ring_finger_tip.x, ring_finger_tip.y), (pinky_tip.x, pinky_tip.y))
                
                # Calculate hand size (distance between thumb tip and pinky tip)
                hand_size = math.dist((thumb_tip.x, thumb_tip.y), (pinky_tip.x, pinky_tip.y))

                sign=''
                
                # Perform gesture recognition logic based on landmark positions and distances
                if thumb_tip.y < index_finger_tip.y and middle_finger_tip.y < ring_finger_tip.y:
                    cv2.putText(frame, "Thumbs Up", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    sign="Thumbs Up"
                elif thumb_tip.y > index_finger_tip.y and middle_finger_tip.y > ring_finger_tip.y:
                    cv2.putText(frame, "Peace", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    sign="peace"  
                elif thumb_tip.y < index_finger_tip.y and middle_finger_tip.y > ring_finger_tip.y:
                    cv2.putText(frame, "Okay", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    sign="Okay"
                elif thumb_tip.y > index_finger_tip.y and middle_finger_tip.y < ring_finger_tip.y:
                    cv2.putText(frame, "Pointing", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    sign="Pointing"
                else:
                    cv2.putText(frame, "No gesture", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                engine.say(sign)
                engine.runAndWait()
        # Display the frame
        cv2.imshow('Hand Gesture Recognition', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()

