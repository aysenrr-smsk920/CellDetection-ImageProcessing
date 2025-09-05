import cv2
import numpy as np
import os

# Klasör yolu
folder_path = r'C:\Users\AYSENUR SIMSEK\Desktop\YZ-VOST-LP\hucre_goruntuleri\histogram'
output_folder = os.path.join(folder_path, "processedHistogram")

# Çıkış klasörü yoksa oluştur
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

#Klasördeki tüm görüntüleri al
image_files = [f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.jpeg', '.png'))]

for image_file in image_files:
    #Görüntüyü oku
    image_path = os.path.join(folder_path, image_file)
    image = cv2.imread(image_path)

    if image is None:
        print(f"Hata: {image_file} yüklenemedi!")
        continue

    #Görüntüyü yeniden boyutlandır (isteğe bağlı)
    scale_factor = 0.3
    image_resized = cv2.resize(image, (int(image.shape[1] * scale_factor), int(image.shape[0] * scale_factor)))

    #Gürültü Azaltma (Bilateral Filter)
    noise_reduced = cv2.bilateralFilter(image_resized, d=9, sigmaColor=75, sigmaSpace=75)

    #Gaussian Blur ile daha fazla yumuşatma
    gaussian_blur = cv2.GaussianBlur(noise_reduced, (5, 5), 0)

    #Kenar Bulma (Canny)
    edges = cv2.Canny(gaussian_blur, threshold1=50, threshold2=150)

    #İşlenmiş görüntüleri kaydet
    noise_reduced_path = os.path.join(output_folder, f"noise_reduced_{image_file}")
    edges_output_path = os.path.join(output_folder, f"edges_{image_file}")

    cv2.imwrite(noise_reduced_path, noise_reduced)
    cv2.imwrite(edges_output_path, edges)

    print(f"{image_file} işlendi ve kaydedildi.")

print("Tüm görüntüler işlendi!")
