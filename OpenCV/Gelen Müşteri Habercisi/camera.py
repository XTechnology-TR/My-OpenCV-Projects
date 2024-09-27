# Gerekli Kütüphaneler
import cv2
import pygame
import numpy as np
import time

# Pygame başlat
pygame.init()

# Ses dosyasını yükle
pygame.mixer.init()
pygame.mixer.music.load('ses.mp3')

# OpenCV'de önceden eğitilmiş Haar Cascade sınıflandırıcısını yükle
human_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_fullbody.xml')

# Kamerayı başlat
cap = cv2.VideoCapture(0)

# Pygame'de ekran boyutu belirle
screen = pygame.display.set_mode((640, 480))

# Ana döngü
running = True
while running:
    ret, frame = cap.read()
    if not ret:
        break

    # Grayscale (gri tonlamalı) görüntüye çevir
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # İnsanları tespit et
    humans = human_cascade.detectMultiScale(gray, 1.1, 4)

    # Her tespit edilen insan için dikdörtgen çiz
    for (x, y, w, h) in humans:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Eğer insan tespit edildiyse ses çal
        if len(humans) > 0:
            # Eğer müzik çalmıyorsa çal
            if not pygame.mixer.music.get_busy():
                pygame.mixer.music.play()
            time.sleep(1)  # Biraz bekleme süresi ekle (farklı insan tespitleri için ses kesintisini önler)

    # OpenCV'nin BGR formatındaki görüntüsünü RGB formatına çevir
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Pygame'de görüntüyü gösterebilmek için OpenCV görüntüsünü Pygame'e uygun hale getir
    frame = np.rot90(frame)  # OpenCV ve Pygame arasında eksen farkını düzelt
    frame = pygame.surfarray.make_surface(frame)

    # Pygame ekranına görüntüyü çiz
    screen.blit(frame, (0, 0))
    pygame.display.update()

    # Pygame'de çıkış olayını kontrol et
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

# Kamerayı serbest bırak ve Pygame'i kapat
cap.release()
pygame.quit()
