import cv2
import numpy as np

# Sabitler (Magic Numbers) tanımlandı
GAUSSIAN_BLUR_KERNEL_SIZE = (5, 5)
CANNY_EDGE_THRESHOLD_1 = 50
CANNY_EDGE_THRESHOLD_2 = 150
HOUGH_LINE_THRESHOLD = 50
MAX_LINE_GAP = 50
LINE_COLOR = (0, 255, 0)
LINE_THICKNESS = 5

def convert_to_grayscale(image):
    """Görüntüyü gri tonlamaya çevirir."""
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def apply_gaussian_blur(image, kernel_size):
    """Görüntüyü bulanıklaştırır."""
    return cv2.GaussianBlur(image, kernel_size, 0)

def detect_edges(image, threshold1, threshold2):
    """Canny algoritmasıyla kenar tespiti yapar."""
    return cv2.Canny(image, threshold1, threshold2)

def detect_lines(image, edges, threshold, max_line_gap):
    """Hough Dönüşümü ile çizgileri tespit eder."""
    return cv2.HoughLinesP(edges, 1, np.pi / 180, threshold, maxLineGap=max_line_gap)

def draw_lines(image, lines, color, thickness):
    """Görüntü üzerinde tespit edilen çizgileri çizer."""
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(image, (x1, y1), (x2, y2), color, thickness)
    return image

def process_image(image):
    """Görüntüyü işler: gri tonlama, bulanıklaştırma, kenar tespiti ve çizgi çizme."""
    gray_image = convert_to_grayscale(image)
    blurred_image = apply_gaussian_blur(gray_image, GAUSSIAN_BLUR_KERNEL_SIZE)
    edges = detect_edges(blurred_image, CANNY_EDGE_THRESHOLD_1, CANNY_EDGE_THRESHOLD_2)
    lines = detect_lines(image, edges, HOUGH_LINE_THRESHOLD, MAX_LINE_GAP)
    return draw_lines(image, lines, LINE_COLOR, LINE_THICKNESS)

def main(video_path):
    """Ana fonksiyon: Video okuma, işleme ve görüntüleme döngüsü."""
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Video dosyası açılamadı!")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Video okuma işlemi bitti veya hata oluştu.")
            break

        processed_frame = process_image(frame)
        cv2.imshow('Şerit Takibi', processed_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):  # 'q' tuşuna basıldığında çık
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    video_path = 'yol.mp4'  # Video dosyası
    main(video_path)
