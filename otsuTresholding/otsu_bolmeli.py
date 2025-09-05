import cv2
import numpy as np
import os

# Kaynak klasör ve kayıt klasörü
source_dir = r'C:\Users\AYSENUR SIMSEK\Desktop\YZ-VOST-LP\hucre_goruntuleri\otsu1\otsu'
save_dir = r'C:\Users\AYSENUR SIMSEK\Desktop\YZ-VOST-LP\hucre_goruntuleri\otsu1\processed_bolmeli'

# Kayıt klasörünü oluştur
os.makedirs(save_dir, exist_ok=True)

# Klasördeki tüm görüntüleri al
image_files = [f for f in os.listdir(source_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

# Bölme boyutları (örnek olarak 2x2)
num_rows = 2
num_cols = 2

for image_file in image_files:
    image_path = os.path.join(source_dir, image_file)
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        print(f"Hata: {image_path} okunamadı! Atlanıyor...")
        continue

    # Görüntü boyutlarını al
    h, w = image.shape
    h_step, w_step = h // num_rows, w // num_cols

    # Parçaları saklamak için liste
    processed_pieces = []

    # Görüntüyü böl ve her parça üzerinde işlem yap
    for i in range(num_rows):
        row_pieces = []
        for j in range(num_cols):
            # Alt bölgeyi seç
            y1, y2 = i * h_step, (i + 1) * h_step
            x1, x2 = j * w_step, (j + 1) * w_step
            piece = image[y1:y2, x1:x2]

            # Otsu thresholding uygula
            _, binary_piece = cv2.threshold(piece, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # İşlenen parçayı listeye ekle
            row_pieces.append(binary_piece)

        # Satırları birleştir
        processed_pieces.append(np.hstack(row_pieces))

    # Tüm satırları birleştirerek tam görüntüyü oluştur
    final_image = np.vstack(processed_pieces)

    # Sonuçları kaydet
    save_path = os.path.join(save_dir, f'otsuBolmeli_{image_file}')
    cv2.imwrite(save_path, final_image)

print("Tüm görüntüler bölündü, işlendi ve tekrar birleştirildi.")
