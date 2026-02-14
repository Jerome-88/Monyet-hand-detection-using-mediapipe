import cv2 
import mediapipe as mp 
import math

mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

reactions = {
    "finger": cv2.imread(r"C:\Users\Jerome\DataScience\gesture_react_image.py-20251015T162136Z-1-001\gesture_react_image.py\react1.png"),  
    "mouth": cv2.imread(r"C:\Users\Jerome\DataScience\gesture_react_image.py-20251015T162136Z-1-001\gesture_react_image.py\react2.png"),   
    "pray": cv2.imread(r"C:\Users\Jerome\DataScience\gesture_react_image.py-20251015T162136Z-1-001\gesture_react_image.py\react3.png"),    
    "phone": cv2.imread(r"C:\Users\Jerome\DataScience\gesture_react_image.py-20251015T162136Z-1-001\gesture_react_image.py\react4.png"),  
    "idle": cv2.imread(r"C:\Users\Jerome\DataScience\gesture_react_image.py-20251015T162136Z-1-001\gesture_react_image.py\react_idle.png"),
    "fucek": cv2.imread(r"C:\Users\Jerome\DataScience\gesture_react_image.py-20251015T162136Z-1-001\gesture_react_image.py\fucek.jpg") 
}

def distance(p1, p2):
    return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

cap = cv2.VideoCapture(0)

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose, \
     mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5, max_num_hands=2) as hands:

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pose_result = pose.process(rgb)
        hands_result = hands.process(rgb)

        gesture = "idle"

        if hands_result.multi_hand_landmarks:
            for hand_landmarks in hands_result.multi_hand_landmarks:
                lm = hand_landmarks.landmark

                wrist = lm[0]
                index_tip = lm[8]
                middle_tip = lm[12]
                thumb_tip = lm[4]

                if distance(index_tip, middle_tip) > 0.1 and distance(index_tip, thumb_tip) > 0.1:
                    gesture = "finger"

                if pose_result.pose_landmarks:
                    nose = pose_result.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE]
                    if distance(index_tip, nose) < 0.15 or distance(thumb_tip, nose) < 0.15:
                        gesture = "mouth"
                    # elif distance(middle_tip, nose) < 0.15:
                    #     gesture="fucek"


        if hands_result.multi_hand_landmarks and len(hands_result.multi_hand_landmarks) == 2:
            hand1 = hands_result.multi_hand_landmarks[0].landmark[0]
            hand2 = hands_result.multi_hand_landmarks[1].landmark[0]
            if distance(hand1, hand2) < 0.5:
                gesture = "pray"

        if hands_result.multi_hand_landmarks:
            hand = hands_result.multi_hand_landmarks[0].landmark[0]
            if 0.3 < hand.x < 0.7 and 0.3 < hand.y < 0.7:
                gesture = "phone"

        reaction_img = reactions.get(gesture, reactions["idle"])
        if reaction_img is None:
            reaction_img = reactions["idle"]

        reaction_img = cv2.resize(reaction_img, (w, h))

        combined = cv2.hconcat([frame, reaction_img])

        if pose_result.pose_landmarks:
            mp_drawing.draw_landmarks(frame, pose_result.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        if hands_result.multi_hand_landmarks:
            for hand_landmarks in hands_result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        cv2.imshow("Gesture Reaction (Gambar)", combined)

        if cv2.waitKey(1) & 0xFF == 27:  
            break

cap.release()
cv2.destroyAllWindows()
