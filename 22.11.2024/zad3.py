import os
from os import listdir
import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
folder_dir = "bird_miniatures"
output_dir = "grayscale_images"
'''zmieanie obrazow na czarno biale'''

for images in os.listdir(folder_dir):
    if (images.endswith(".jpg")):
        print(images)
        image_path = os.path.join(folder_dir, images)
        image = cv2.imread(image_path)

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width, _ = image_rgb.shape
        gray_image = np.zeros((height, width), dtype=np.uint8)

        for y in range(height):
            for x in range(width):
                r, g, b = image_rgb[y, x]  
                gray_value = int(0.299 * r + 0.587 * g + 0.114 * b)
                gray_image[y, x] = gray_value 
        pixels = gray_image.flatten()
        pixel_counts = Counter(pixels)
        most_common_pixel_value, _ = pixel_counts.most_common(1)[0]
        sky_threshold = most_common_pixel_value-15
        sky_image = np.where(gray_image >= sky_threshold, 255, 0).astype(np.uint8)
        final_image = np.ones_like(gray_image) * 255
        final_image[sky_image == 0] = 0
       # distinct_image = np.copy(gray_image)
        #dilated_image = cv2.dilate(black_pixels.astype(np.uint8), kernel, iterations=1)
        #distinct_image[distinct_image > threshold-20] = 0  # Change black to white
          # 5x5 kernel for dilation
        #distinct_image[distinct_image < threshold] = 255  # Change black to white
        #distinct_image[distinct_image > threshold-20] = 0  # Change black to white
        #distinct_image = gray_image.copy()
        #distinct_image[final_image > 0] = 255
        output_path = os.path.join(output_dir, f"grayscale_{images}")  

        birds = 0
        black_pixel = 0
        for y in range(1, height): 
            for x in range(1, width):
                if final_image[y, x] == black_pixel:
                    if final_image[y-1, x] != black_pixel and final_image[y, x-1] != black_pixel:
                        birds += 1
        print(f"Number of birds detected: {birds}")

