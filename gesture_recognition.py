import cv2
import mediapipe as mp
import math

# Inisialisasi mediapipe hand
mp_hands = mp.solutions.hands
hands= mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

# Fungsi untuk mengenal gestur tangan
def recognize_gesture(hand_landmarks):
    # Ambil TIP dari jari
    tip_jempol = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    tip_telunjuk = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    tip_tengah = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    tip_manis = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
    tip_kelingking = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]

    # Ambil PIP/MCP dari jari
    mcp_jempol = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP]
    pip_telunjuk = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP]
    pip_tengah = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP]
    pip_manis = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP]
    pip_kelingking = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP]
    
    # Hitung angle per jari
    angle_thumb = calculate_angle(
        hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC],
        hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP],
        hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    )

    angle_index = calculate_angle(
        hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP],
        hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP],
        hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    )

    angle_middle = calculate_angle(
        hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP],
        hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP],
        hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    )

    angle_ring = calculate_angle(
        hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP],
        hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP],
        hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
    )

    angle_pinky = calculate_angle(
        hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP],
        hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP],
        hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
    )

    # GESTURE : METAL
    if (angle_thumb > 160 and
        angle_index > 160 and
        angle_middle < 160 and
        angle_ring < 160 and
        angle_pinky > 160):
        return "Metal"

    # GESTURE : THUMBS UP & DOWN
    if (angle_thumb > 160 and
        angle_index < 90 and
        angle_middle < 90 and
        angle_ring < 90 and
        angle_pinky < 90):

        if tip_jempol.y < mcp_jempol.y:
            return "Thumbs Up"
        else:
            return 'Thumbs Down'
        
    # GESTURE : PEACE
    if (angle_thumb < 160 and
        angle_index > 160 and
        angle_middle > 160 and
        angle_ring < 160 and
        angle_pinky < 160):
        return "Peace"
    
    # GESTURE : OK SIGN
    dist = math.sqrt(
        (tip_jempol.x - tip_telunjuk.x) ** 2 +
        (tip_jempol.y - tip_telunjuk.y) ** 2 +
        (tip_jempol.z - tip_telunjuk.z) ** 2
    )

    if (dist < 0.05 and
        angle_middle > 150 and
        angle_pinky > 150 and
        angle_ring > 150):
        return "OK Sign"
    
    # GESTURE : CALL ME
    if (angle_thumb > 160 and
        angle_index < 160 and
        angle_middle < 160 and
        angle_ring < 160 and
        angle_pinky > 160):
        return "Call Me" 
    
    # GESTURE : GUN
    if (angle_thumb > 160 and
        angle_index > 160 and
        angle_middle < 160 and
        angle_ring < 160 and
        angle_pinky < 160):
        return "GUN"
    
    if (angle_index < 90 and
        angle_middle < 90 and
        angle_ring < 90 and
        angle_pinky < 90):
        return "Fist"
    
    return "Gesture tidak diketahui"

# Fungsi untuk mendeteksi tangan dengan mediapipe
def detect_hand_gesture(image,hand):
    image_rgb = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    results = hand.process(image_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            gesture = recognize_gesture(hand_landmarks)
            mp_draw.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Ambil posisi wrist
            wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
            h, w, c = image.shape
            cx, cy = int(wrist.x * w), int(wrist.y * h) + 30

            # Menampilkan teks gesture di bawah tangan dengan background
            text = gesture
            font_scale = 1
            thickness = 2
            color_text = (255, 255, 255)
            color_bg = (0, 0, 0)

            # Hitung ukuran text
            (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)

            rect_x1 = cx
            rect_y1 = cy - text_height - baseline
            rect_x2 = cx + text_width
            rect_y2 = cy + baseline

            # Tampilkan text
            cv2.rectangle(image, (rect_x1, rect_y1), (rect_x2, rect_y2), color_bg, -1)
            cv2.putText(image, text, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color_text, thickness, cv2.LINE_AA)

    return image

# Fungsi untuk menghitung angle
def calculate_angle(a, b, c):
    ba = [a.x - b.x, a.y - b.y, a.z - b.z]
    bc = [c.x - b.x, c.y - b.y, c.z - b.z]

    dot_product = ba[0]*bc[0] + ba[1]*bc[1] + ba[2]*bc[2]
    norm_ba = math.sqrt(ba[0]**2 + ba[1]**2 + ba[2]**2)
    norm_bc = math.sqrt(bc[0]**2 + bc[1]**2 + bc[2]**2)

    angle = math.acos(dot_product / (norm_ba * norm_bc))
    return math.degrees(angle)


# Membuka kamera
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("tidak dapat membuka kamera")
    exit()

while(cap.isOpened):
    ret,frame = cap.read()
    if not ret:
        print("Gagal menangkap frame")
        break

    frame = detect_hand_gesture(frame,hands)
    cv2.imshow("Hand gesture Recognition dengan Mediapipe dan OpenCV",frame)

    if cv2.waitKey(1) & 0xFF ==ord('q'):
        break

cap.release()
cv2.destroyAllWindows()