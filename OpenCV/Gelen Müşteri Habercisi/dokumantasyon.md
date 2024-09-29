- Öncelikle projenin amacından bahsetmeliyim,  ben bazen dükkan dışından biraz uzaklaştığımda dükkana giren müşteriler oluyordu ben de geç fark ediyordum.
- Bu durumun önüne geçmek için elimdeki imkanları kullanarak varolan dahili kameram ve biraz kodlama ile müşterilerin dükkan içerisine geldiği anda kamera tespit edip hoparlörden istediğim bir sesi çalıyor, bu sayede hemen dükkana gidip müşterilerle ilgilenebiliyorum.
- Bunun yanısıra dükkanımın yoğunluk saatleri, satılan ürünlerin istatistiği, müşteri yaş istatistiği ve cinsiyeti gibi parametreleri de yakın zamanda yeni versiyonda getirmeyi planlıyorum.
- KVKK kuralları göz önünde bulundurulmuştur.

# Gereksinimler

cihazda python kurulu olmalı. yoksa buradan indirmelisin https://www.python.org/downloads/

kütüphane kurulumlarını gerçekleştir. “pip install …”

kodu exe formatına dönüştürmedim. Eğer hemen kullanılacaksa python to exe yapılmalı.

# Kod Algoritması

### Algoritma

1. **Kütüphaneleri İçe Aktar**
    - `cv2`: Görüntü işleme için OpenCV kütüphanesi.
    - `pygame`: Oyun ve grafik arayüzü oluşturmak için Pygame kütüphanesi.
    - `numpy`: Dizi işlemleri için NumPy kütüphanesi.
    - `time`: Zamanla ilgili işlemler için.
    - `datetime`: Tarih ve saat ile ilgili işlemler için.
2. **Pygame Penceresini Başlat**
    - Pygame başlatılır ve pencere boyutu ayarlanır.
    - Ses modülü başlatılır.
3. **Ses Dosyasını Yükle**
    - Çalınacak ses dosyası (`ses.mp3`) yüklenir.
4. **Kamerayı Başlat**
    - Bilgisayarın kamerası açılır.
5. **Haar Cascade Sınıflandırıcısını Yükle**
    - İnsanları tespit etmek için Haar Cascade sınıflandırıcısı (`haarcascade_fullbody.xml`) yüklenir.
6. **Ana Döngü**
    - Sonsuz bir döngüde aşağıdaki adımlar tekrarlanır:
    1. **Kameradan Görüntü Al**
        - Kamera görüntüsü alınır.
    2. **Görüntüyü İşle**
        - Görüntü gri tonlamalı hale getirilir.
        - Renkli görüntü Pygame için uygun hale getirilir (RGB formatına dönüştürülür ve döndürülür).
    3. **İnsan Tespit Et**
        - Gri tonlamalı görüntüde insanları tespit etmek için Haar Cascade sınıflandırıcısı kullanılır.
    4. **İnsanları Çiz**
        - Eğer insanlar tespit edilirse, tespit edilenlerin etrafına dikdörtgen çizilir.
    5. **İnsan Sayısını Kontrol Et**
        - Tespit edilen insan sayısı kontrol edilir.
        - Eğer 1 veya daha fazla insan tespit edilirse:
            - Ses dosyası çalınır.
            - İnsan tespiti bir log dosyasına kaydedilir (tarih ve saat ile birlikte).
            - Sesin tekrar çalmaması için 1 saniye beklenir.
    6. **Pygame Ekranına Görüntüyü Çiz**
        - İşlenmiş görüntü Pygame penceresine çizilir.
    7. **Tespit Edilen İnsan Sayısını Göster**
        - Pygame penceresinde tespit edilen insan sayısı gösterilir.
    8. **Çıkış Olaylarını Kontrol Et**
        - Kullanıcının pencereyi kapatma isteği kontrol edilir. Eğer pencere kapatılmak istenirse, döngüden çıkılır.
7. **Kaynakları Serbest Bırak**
    - Kamera kapatılır.
    - Pygame kapanır.

# ***Kod İçeriği***

```python
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

if **name** == '__main__':
main()
```

## 1.***AŞAMA***

Python’da bazen uzunca kodlar yazmak yerine gerekli kütüphaneleri kullanarak kod duruluğunu sağlarız. Bu kodda ihtiyacımız olan kütüphaneler cv2, pygame, numpy, time, datetime.  Peki neden gerekli?

- `cv2`: OpenCV kütüphanesi, görüntü işleme işlemleri için kullanılır.
- `pygame`: Grafik ve ses işlemlerini yönetmek için Pygame kütüphanesi kullanılır.
- `numpy`: OpenCV görüntülerini Pygame ile uyumlu hale getirmek için gereklidir.
- `time`: Ses çalındıktan sonra bekleme işlemleri için kullanılır.
- `datetime`: Tespit edilen insanların log kayıtlarını tarih ve saatle kaydetmek için kullanılır.

## ***2.AŞAMA***

İşlevsellik açısından pygame ile kodumuza başlamak en mantıklı ve optimize olanı olacaktır.

```python
def initialize_pygame_window(window_size=(640, 480)):
pygame.init()
screen = pygame.display.set_mode(window_size)
pygame.mixer.init()
return screen
```

> def= python’da fonksiyon oluşturmak için kullanırız.
> 
- `pygame.init()`: Pygame'in genel başlangıcını yapar. (init başlama komutu.)
- `pygame.display.set_mode(window_size)`: Boyutu `640x480` olan bir pencere oluşturur. (istersen değişebilirsin.)
- `pygame.mixer.init()`: Ses çalmak için Pygame’in ses motorunu başlatır.

## 3. Ses Dosyasını Yükleme

```python
def load_sound(sound_file='ses.mp3'):
pygame.mixer.music.load(sound_file)
```

### 4. Ses Çalma

```python
def play_sound():
    if not pygame.mixer.music.get_busy():
        pygame.mixer.music.play()
```

Bu fonksiyon, sesin çalınıp çalınmadığını kontrol eder. Eğer ses çalmıyorsa, `play()` fonksiyonu ile çalınır…

Buradan sonra kamerayı kullanmamız gerekiyor. ses ile ilgili işleri hallettik sayılır.

## 5. Kamerayı Başlatma

```python
def initialize_camera():
    return cv2.VideoCapture(0)
```

 OpenCV’nin `VideoCapture` sınıfını kullanarak video akışını alır. 
Buradaki 0 ın değeri eğer harici yani başka bir kamera kullanıyorsan 1 veya 2 de olabilir. Biraz deneyip görmen lazım. Benim dahili kameram olduğu için 0 uygundur.

## 6. İnsan Tespit Modülünü Yükleme

```python
def load_human_cascade():
    return cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_fullbody.xml')
```

bu kısmı internetten araştırıp bulmuştum. şu şekilde açıklanabilir: OpenCV'nin Haar Cascade sınıflandırıcısı kullanılarak insan tespiti için gerekli modeli yükler. Bu model, tam vücut tespiti yapmak için eğitilmiştir.

### 7. İnsanları Tespit Etme

```python
def detect_humans(cascade, gray_frame):
    return cascade.detectMultiScale(gray_frame, 1.1, 4)
```

 gri tonlamalı bir görüntüde insanları tespit eder, bunu yapmamızın sebebi görüntüyü okumayı daha kolay yapması.

- `detectMultiScale()`: Belirtilen görüntüde (ölçek ve adım parametreleri ile) nesneleri tespit eder. (ölçeklemelerin en uygununun bu olduğunu gördüm. İnternetten bulmuştum.

## 8. Tespit Edilen İnsanların Üzerine Dikdörtgen Çizme

```python
def draw_human_boxes(frame, humans):
    for (x, y, w, h) in humans:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
```

Tespit edilen insanların etrafına bir dikdörtgen çizer. `cv2.rectangle()` fonksiyonu, görüntünün üzerine belirlenen koordinatlarla kırmızı bir dikdörtgen çizer.

## 9. Görüntü İşleme

```python
def process_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rotated_frame = np.rot90(rgb_frame)
    return gray, rotated_frame

```

Bu fonksiyon görüntüyü iki şekilde işler:

- `gray`: İnsan tespiti için gerekli gri tonlamalı görüntüyü elde eder.
- `rotated_frame`: Görüntüyü Pygame'e uygun hale getirir (RGB formatında ve 90 derece döndürülmüş).  (ezbere bir kod bu, internetten bulmuştum.

## 10. Log Kaydı Oluşturma

```python
def log_detection(human_count):
    with open('human_detection_log.txt', 'a') as log_file:
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_file.write(f'{timestamp} - Tespit edilen insan sayisi: {human_count}\n')

```

zamanlı biçimde geçmiş müşterilerin tablosunu çizmemizi sağlar ve dosyaya kaydeder.

## 11. İnsan Sayısını Gösterme

```python
def display_human_count(screen, count):
    font = pygame.font.Font(None, 36)
    text = font.render(f'İnsan Sayısı: {count}', True, (255, 255, 255))
    screen.blit(text, (10, 10))

```

Ekranda tespit edilen insan sayısını gösterir. `pygame.font.Font()` ile bir yazı tipi ve boyut belirlenir, `render()` ile metin oluşturulur ve `blit()` ile ekranda belirtilen koordinatlara çizilir.

## 12. Ana Döngü ve Genel Akış

```python
def main():
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
                time.sleep(1)

        screen.blit(pygame.surfarray.make_surface(pygame_frame), (0, 0))
        display_human_count(screen, human_count)
        pygame.display.update()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

    cap.release()
    pygame.quit()

```

Bu kısımda kodun ana akışıdır.:

1. Pygame penceresi açılır ve ses dosyası yüklenir.
2. Kamera başlatılır ve tespit edilecek model yüklenir.
3. Döngü içinde:
    - Kameradan gelen görüntü işlenir.
    - İnsanlar tespit edilir ve sayıları ekranda gösterilir.
    - Eğer insan sayısı pozitifse ses çalınır ve log dosyasına kaydedilir.
    - Ekrana görüntü ve insan sayısı çizilir.
4. Pygame penceresinden çıkıldığında döngü sona erer, kamera serbest bırakılır ve Pygame kapatılır.
