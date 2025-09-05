import cv2
import numpy as np
import os

# Klasör yolu
folder_path = r'C:\Users\AYSENUR SIMSEK\Desktop\YZ-VOST-LP\hucre_goruntuleri\contrast\processedCanny'
output_folder = os.path.join(folder_path, "processedCannyDelDol")

# Çıkış klasörü yoksa oluştur
os.makedirs(output_folder, exist_ok=True)

# Klasördeki tüm görüntüleri al
image_files = [f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.jpeg', '.png'))]

for image_file in image_files:
    # Görüntüyü oku (Canny zaten uygulanmış)
    image_path = os.path.join(folder_path, image_file)
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        print(f"Hata: {image_file} yüklenemedi!")
        continue

    # Konturları Bul
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Yeni siyah bir görüntü oluştur (aynı boyutta)
    filled_contours = np.zeros_like(image)

    # Konturların içini doldur
    cv2.drawContours(filled_contours, contours, -1, (255, 255, 255), thickness=cv2.FILLED)

    # Morfolojik İşlemler (Daha Doğal Şekil İçin)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))  # Eliptik kernel
    closed = cv2.morphologyEx(filled_contours, cv2.MORPH_CLOSE, kernel)

    # Delik Doldurma (Flood Fill)
    filled = closed.copy()
    h, w = filled.shape
    mask = np.zeros((h+2, w+2), np.uint8)  # Flood fill için maske

    # Sadece hücre içlerini doldurması için flood fill uygulayacağız
    for cnt in contours:
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            cv2.floodFill(filled, mask, (cx, cy), 255)

    # Doldurulan Alanları Birleştir
    filled_inv = cv2.bitwise_not(filled)
    final_result = closed | filled_inv

    # Sonucu kaydet
    output_path = os.path.join(output_folder, f"kiv_cekcik_deldol_{image_file}")
    cv2.imwrite(output_path, final_result)

    print(f"{image_file} işlemi tamamlandı ve kaydedildi.")

print("Tüm görüntüler başarıyla işlendi!")
