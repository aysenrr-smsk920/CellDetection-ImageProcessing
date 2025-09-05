import cv2
import numpy as np
import os

# Klasör yolu
folder_path = r'C:\Users\AYSENUR SIMSEK\Desktop\YZ-VOST-LP\hucre_goruntuleri\contrast'
output_folder = os.path.join(folder_path, "processedCanny")

# Çıkış klasörü yoksa oluştur
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Klasördeki tüm görüntüleri al
image_files = [f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.jpeg', '.png'))]

for image_file in image_files:
    # Görüntüyü oku
    image_path = os.path.join(folder_path, image_file)
    image = cv2.imread(image_path)

    if image is None:
        print(f"Hata: {image_file} yüklenemedi!")
        continue

    # Görüntüyü yeniden boyutlandır (isteğe bağlı)
    scale_factor = 0.3  # İstediğiniz oranı ayarlayabilirsiniz
    image_resized = cv2.resize(image, (int(image.shape[1] * scale_factor), int(image.shape[0] * scale_factor)))

    # Gaussian Blur Uygula
    gaussian_blur = cv2.GaussianBlur(image_resized, (5, 5), 0)

    # Canny Kenar Tespiti
    edges = cv2.Canny(image_resized, threshold1=50, threshold2=150)

    # Kenar tespitinin sonuçlarını kaydet
    canny_output_path = os.path.join(output_folder, f"canny_{image_file}")
    cv2.imwrite(canny_output_path, edges)

    print(f"{image_file} işlendi ve kaydedildi.")

print("Tüm görüntüler işlendi!")
