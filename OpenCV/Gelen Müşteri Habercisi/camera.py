import cv2
import pygame
import numpy as np
import time
from datetime import datetime

def initialize_pygame_window(window_size=(640, 480)):
    """Pygame penceresini başlat ve ekran boyutunu ayarla."""
    pygame.init()
    screen = pygame.display.set_mode(window_size)
    pygame.mixer.init()
    return screen

def load_sound(sound_file='ses.mp3'):
    """Ses dosyasını yükle."""
    pygame.mixer.music.load(sound_file)

def play_sound():
    """Ses dosyasını çal."""
    if not pygame.mixer.music.get_busy():
        pygame.mixer.music.play()

def initialize_camera():
    """Kamerayı başlat."""
    return cv2.VideoCapture(0)

def load_human_cascade():
    """İnsan tespiti için Haar Cascade sınıflandırıcısını yükle."""
    return cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_fullbody.xml')

def detect_humans(cascade, gray_frame):
    """Gri tonlamalı görüntüde insanları tespit et."""
    return cascade.detectMultiScale(gray_frame, 1.1, 4)

def draw_human_boxes(frame, humans):
    """Tespit edilen insanların etrafına dikdörtgen çiz."""
    for (x, y, w, h) in humans:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

def process_frame(frame):
    """Görüntüyü gri tonlamalıya çevir ve OpenCV'den Pygame'e uygun hale getir."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rotated_frame = np.rot90(rgb_frame)
    return gray, rotated_frame

def log_detection(human_count):
    """İnsan tespiti olduğunda log dosyasına kaydedin."""
    with open('human_detection_log.txt', 'a') as log_file:
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_file.write(f'{timestamp} - Tespit edilen insan sayisi: {human_count}\n')

def display_human_count(screen, count):
    """Tespit edilen insan sayısını ekranda göster."""
    font = pygame.font.Font(None, 36)
    text = font.render(f'İnsan Sayısı: {count}', True, (255, 255, 255))
    screen.blit(text, (10, 10))

def main():
    """Ana döngü, insan tespiti, ses çalma, loglama ve insan sayısını gösterme işlemlerini yönetir."""
    screen = initialize_pygame_window()
    load_sound()
    cap = initialize_camera()
    human_cascade = load_human_cascade()

    running = True
    while running:
        ret, frame = cap.read()
        if not ret:
            break

        gray_frame, pygame_frame = process_frame(frame)
        humans = detect_humans(human_cascade, gray_frame)

        if humans is not None:
            draw_human_boxes(frame, humans)
            human_count = len(humans)

            if human_count > 0:
                play_sound()
                log_detection(human_count)
                time.sleep(1)  # Sesin tekrar çalmaması için bekleme

        # Pygame ekranına görüntüyü çiz
        screen.blit(pygame.surfarray.make_surface(pygame_frame), (0, 0))

        # Tespit edilen insan sayısını ekranda göster
        display_human_count(screen, human_count)

        pygame.display.update()

        # Pygame'de çıkış olayını kontrol et
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

    cap.release()
    pygame.quit()

if __name__ == '__main__':
    main()
