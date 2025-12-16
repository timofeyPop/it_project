import cv2# OpenCV для работы с видео
import mediapipe as mp# Для отслеживания положения рук
import random # Для случайного выбора ИИ
import time
from collections import Counter, deque
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import math

def angle(a, b, c):
    # угол ABC (в градусах), a,b,c — (x,y,z) или (x,y)
    ab = [a[i]-b[i] for i in range(len(a))]
    cb = [c[i]-b[i] for i in range(len(a))]
    dot = sum(ab[i]*cb[i] for i in range(len(a)))
    nab = math.sqrt(sum(ab[i]*ab[i] for i in range(len(a))))
    ncb = math.sqrt(sum(cb[i]*cb[i] for i in range(len(a))))
    cosv = dot / (nab*ncb + 1e-9)
    cosv = max(-1.0, min(1.0, cosv))
    return math.degrees(math.acos(cosv))

# Инициализация MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

FINGERS = {
    "index":  (5, 6, 7, 8),
    "middle": (9,10,11,12),
    "ring":   (13,14,15,16),
    "pinky":  (17,18,19,20),
}

# Очередь для сглаживания показаний пальцев
gesture_buffer = deque(maxlen=5)

# Возможные жесты
gestures = {"?": 0, "Камень": 1, "Ножницы": 2, "Бумага": 3}

# Открытие камеры
cap = cv2.VideoCapture(0)

# Загрузка шрифта (путь может отличаться)
font_path = "arial.ttf"  # Нужно, чтобы поддерживалась кириллица
font = ImageFont.truetype(font_path, 32)

# Функция для рисования текста
def draw_text(img, text, position, font, color=(255, 255, 255)):
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    draw.text(position, text, font=font, fill=color)
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

def is_finger_bent(lm, finger_name, pip_thr=160, dip_thr=160):
    mcp, pip, dip, tip = FINGERS[finger_name]
    def p(i):
        return (lm[i].x, lm[i].y, lm[i].z)

    ang_pip = angle(p(mcp), p(pip), p(dip))  # MCP-PIP-DIP
    ang_dip = angle(p(pip), p(dip), p(tip))  # PIP-DIP-TIP

    # если хотя бы один сустав “сломался” достаточно сильно — считаем палец согнут
    return (ang_pip < pip_thr) or (ang_dip < dip_thr), (ang_pip, ang_dip)

# Таймеры
game_start_time = time.time()  # Время начала раунда
round_duration = 3  # Секунды на выбор жеста

player_choice = None
ai_choice = None
result_text = "Сделай свой выбор!"

with mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7, max_num_hands=1) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = hands.process(rgb_frame)

        # Таймер
        elapsed_time = time.time() - game_start_time

        # Определение выбора игрока
        if elapsed_time < round_duration:
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                    unbented_fingers={"index":  1,
                                    "middle": 1,
                                    "ring":   1,
                                    "pinky":  1}
                    
                    for finger in FINGERS.keys():
                        if is_finger_bent(hand_landmarks.landmark, finger)[0]:
                            unbented_fingers[finger] = 0

                    # Подсчет пальцев
                    if sum(unbented_fingers.values())==0:                                                                gesture_buffer.append(1)
                    elif sum(unbented_fingers.values())==2 and unbented_fingers["index"] and unbented_fingers["middle"]: gesture_buffer.append(2)
                    elif sum(unbented_fingers.values())==4:                                                              gesture_buffer.append(3)
                    else:                                                                                                gesture_buffer.append(0)

                    # Сглаживание
                    stable_count = Counter(gesture_buffer).most_common(1)[0][0]

                    # Фиксация выбора игрока
                    player_choice = list(gestures.keys())[stable_count]

            remaining_time = round(round_duration - elapsed_time, 1)
            timer_text = f"Выбор через: {remaining_time} сек"
        else:
            timer_text = "Выбор завершён!"

            # Фиксация выбора ИИ только один раз
            if ai_choice is None and player_choice:
                ai_choice = random.choice(list(gestures.keys())[1:])

                # Определение победителя
                if player_choice == ai_choice:
                    result_text = "Ничья!"
                elif (player_choice == "Камень" and ai_choice == "Ножницы") or \
                     (player_choice == "Ножницы" and ai_choice == "Бумага") or \
                     (player_choice == "Бумага" and ai_choice == "Камень"):
                    result_text = "Ты выиграл!"
                else:
                    result_text = "ИИ выиграл!"

            # Перезапуск раунда
            if elapsed_time > round_duration + 2:
                game_start_time = time.time()  # Сброс таймера
                player_choice = None
                ai_choice = None
                result_text = "Сделай свой выбор!"

        # Вывод информации
        frame = draw_text(frame, timer_text, (50, 50), font, (0, 255, 255))
        frame = draw_text(frame, f'Ты: {player_choice if player_choice else "?"}', (50, 100), font, (0, 255, 0))
        frame = draw_text(frame, f'ИИ: {ai_choice if ai_choice else "?"}', (50, 150), font, (255, 0, 0))
        frame = draw_text(frame, result_text, (50, 200), font, (255, 255, 0))

        cv2.imshow('ROCK-PAPER-SCISSORS', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
