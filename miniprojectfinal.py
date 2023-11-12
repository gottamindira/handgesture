import cv2
import mediapipe as mp

# Initialize Mediapipe hands module
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Initialize webcam
cap = cv2.VideoCapture(0)

with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            continue

        # Convert BGR image to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process image with Mediapipe hands
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)

                # Calculate thumb and index finger positions
                thumb_tip = landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                index_finger_tip = landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

                # Calculate Euclidean distance between thumb and index finger tips
                distance = ((thumb_tip.x - index_finger_tip.x) * 2 + (thumb_tip.y - index_finger_tip.y) * 2) ** 0.5

                # Detect thumbs-up gesture based on distance
                if distance < 0.05:
                    cv2.putText(frame, "Thumbs Up", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # Detect peace sign (V shape) gesture based on distance
                if distance < 0.05:
                    cv2.putText(frame, "Peace Sign", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display the frame
        cv2.imshow("Hand Gesture Detection", frame)

        if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to exit
            break

cap.release()
cv2.destroyAllWindows()
