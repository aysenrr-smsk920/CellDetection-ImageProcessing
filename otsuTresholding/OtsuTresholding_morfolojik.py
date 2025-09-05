import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# Klasör yolları
input_folder = r'C:\Users\AYSENUR SIMSEK\Desktop\YZ-VOST-LP\hucre_goruntuleri\contrast\processedCanny'
output_folder = os.path.join(input_folder, 'OtsuMorfolojik')

# Çıktı klasörünü oluştur
os.makedirs(output_folder, exist_ok=True)

# Tüm .jpeg dosyalarını bul
image_files = [f for f in os.listdir(input_folder) if f.endswith('.jpeg')]

for image_file in image_files:
    image_path = os.path.join(input_folder, image_file)
    edge_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Kenarları genişletme (dilate)
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(edge_image, kernel, iterations=2)

    # İç bölgeleri doldurma işlemi için threshold uygulama
    _, binary = cv2.threshold(dilated, 50, 255, cv2.THRESH_BINARY)

    # Kenarları bulma
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # İçleri doldurulmuş yeni bir görüntü oluşturma
    filled_image = np.zeros_like(edge_image)
    cv2.drawContours(filled_image, contours, -1, (255), thickness=cv2.FILLED)

    # İşlenmiş görüntüyü kaydet
    output_path = os.path.join(output_folder, f'otsumorf_{image_file}')
    cv2.imwrite(output_path, filled_image)

    print(f"İşlenmiş görüntü kaydedildi: {output_path}")

print("Tüm görüntüler işlendi ve kaydedildi!")
