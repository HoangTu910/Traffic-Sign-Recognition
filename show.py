from PIL import Image
import matplotlib.pyplot as plt

def display_ppm_image(image_path):
    # Mở hình ảnh sử dụng PIL
    image = Image.open(image_path)

    # Hiển thị hình ảnh sử dụng matplotlib
    plt.imshow(image)
    plt.axis('off')  # Ẩn trục tọa độ
    plt.show()

# Đường dẫn đến file PPM
image_path = 'gtsrb/40/00003_00000.ppm'

display_ppm_image(image_path)
