import cv2
import numpy as np
import os

# Kaynak ve hedef klasörleri belirleme
source_dir = r'C:\Users\AYSENUR SIMSEK\Desktop\YZ-VOST-LP\hucre_goruntuleri\otsu1\otsu'
save_dir = r'C:\Users\AYSENUR SIMSEK\Desktop\YZ-VOST-LP\hucre_goruntuleri\otsu1\processedotsu'

# Kayıt klasörünü oluştur (eğer yoksa)
os.makedirs(save_dir, exist_ok=True)

# Klasördeki tüm görüntü dosyalarını al
image_files = [f for f in os.listdir(source_dir) if f.endswith(('.jpeg', '.png'))]

for image_file in image_files:
    # Görüntüyü yükleme
    image_path = os.path.join(source_dir, image_file)
    edge_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Otsu thresholding uygulama
    _, binary = cv2.threshold(edge_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Kenarları bulma ve içlerini doldurma
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled_image = np.zeros_like(edge_image)
    cv2.drawContours(filled_image, contours, -1, (255), thickness=cv2.FILLED)

    # Yeni isimlerle kayıt etme
    binary_save_path = os.path.join(save_dir, f'otsu_{image_file}')
    filled_save_path = os.path.join(save_dir, f'filled_{image_file}')

    cv2.imwrite(binary_save_path, binary)
    #cv2.imwrite(filled_save_path, filled_image)

print("İşlem tamamlandı, tüm görüntüler 'processedotsu' klasörüne kaydedildi.")
