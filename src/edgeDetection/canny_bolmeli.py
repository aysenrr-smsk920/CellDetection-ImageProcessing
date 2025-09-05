import cv2
import numpy as np
import os

# Klasör yolu
folder_path = r'C:\Users\AYSENUR SIMSEK\Desktop\YZ-VOST-LP\hucre_goruntuleri\contrast'
output_folder = os.path.join(folder_path, "processedCannyBolmeli")

# Çıkış klasörü yoksa oluştur
os.makedirs(output_folder, exist_ok=True)

# Klasördeki tüm görüntüleri al
image_files = [f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.jpeg', '.png'))]

# Kaç parçaya bölünecek? (Örn: 2x2)
grid_size = (2, 2)  # 2 satır, 2 sütun

for image_file in image_files:
    # Görüntüyü oku (Renkli)
    image_path = os.path.join(folder_path, image_file)
    image = cv2.imread(image_path)

    if image is None:
        print(f"Hata: {image_file} yüklenemedi!")
        continue

    # Görüntüyü yeniden boyutlandır (isteğe bağlı)
    scale_factor = 0.3  # Görüntü boyutunu %30'a düşürme
    image_resized = cv2.resize(image, (int(image.shape[1] * scale_factor), int(image.shape[0] * scale_factor)))

    # Görüntü boyutları
    h, w, _ = image_resized.shape
    gh, gw = grid_size  # Grid bölme sayıları (2x2)

    # Parçaların boyutları
    h_part, w_part = h // gh, w // gw

    # Boş bir görüntü (aynı boyutta, birleşme için)
    stitched_image = np.zeros_like(image_resized)

    for i in range(gh):
        for j in range(gw):
            # Parçayı kes
            y1, y2 = i * h_part, (i + 1) * h_part
            x1, x2 = j * w_part, (j + 1) * w_part
            image_part = image_resized[y1:y2, x1:x2]

            # Gaussian Blur Uygula (Gürültüyü azaltmak için)
            blurred = cv2.GaussianBlur(image_part, (5, 5), 0)

            # Renkli görüntüde Canny uygulamak için her kanala ayrı kenar tespiti
            edges_b = cv2.Canny(blurred[:, :, 0], 50, 150)  # Mavi Kanal
            edges_g = cv2.Canny(blurred[:, :, 1], 50, 150)  # Yeşil Kanal
            edges_r = cv2.Canny(blurred[:, :, 2], 50, 150)  # Kırmızı Kanal

            # Üç kanalın maksimum değerini alarak birleşik kenar görüntüsü oluştur
            edges_combined = np.maximum(edges_b, np.maximum(edges_g, edges_r))

            # Sonucu birleşik görüntüye ekle
            stitched_image[y1:y2, x1:x2] = cv2.merge([edges_combined] * 3)  # Renkli halde kaydetmek için

    # Kenar tespitinin sonuçlarını kaydet
    canny_output_path = os.path.join(output_folder, f"cannybolmeli_{image_file}")
    cv2.imwrite(canny_output_path, stitched_image)

    print(f"{image_file} bölmeli olarak işlendi ve kaydedildi.")

print("Tüm görüntüler bölmeli olarak işlendi!")
