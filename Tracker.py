import cv2
import mediapipe as mp
import time

# initialize video capture from default camera
cap = cv2.VideoCapture(0)

# initialize hand detection module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,  # Set to True to improve detection quality, but slower fps
    max_num_hands=2,  # Detect up to 2 hands in the frame
    min_detection_confidence=0.7,  # Set minimum confidence threshold for detection
    min_tracking_confidence=0.5)  # Set minimum confidence threshold for tracking

# initialize drawing utility
mp_draw = mp.solutions.drawing_utils

# initialize timer variables
current_time = 0
previous_time = 0

# start the video capture loop
while True:
    
    # read the frame from the video capture
    success, frame = cap.read()

    if not success:
        print("Failed to read video capture")
        break

    # convert the color space from BGR to RGB for media pipe
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # process the image with hand detection module
    results = hands.process(image_rgb)

    # if hands detected, draw landmarks on the frame
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # draw a circle around the tip of index finger
            index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            cx, cy = int(index_finger_tip.x * frame.shape[1]), int(index_finger_tip.y * frame.shape[0])
            cv2.circle(frame, (cx, cy), 15, (255, 0, 255), 6)

    # calculate frames per second and display it on the frame
    current_time = time.time()
    fps = 1 / (current_time - previous_time)
    previous_time = current_time
    cv2.putText(frame, f"FPS: {int(fps)}", (10, 70), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 255), 2)

    # display the resulting frame
    cv2.imshow("Hand Detection", frame)

    # exit the loop when "q" is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# release resources and close windows
hands.close()
cap.release()
cv2.destroyAllWindows()
