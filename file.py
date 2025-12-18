import pydicom as dicom
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

# зургаа уншихдаа байгаа газраас нь дуудна. Заавал нормлайз хийнэ.
dicom_data=dicom.dcmread ("./img/sample2.dcm")
img=dicom_data.pixel_array
img=cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
# thresholding hiine-contour oloh
ret, binary=cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)
# contourlah
contours, hierarchy=cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#"одоо зургаа хадгалах "
def save_image(image, output_path):
        # ur dung haruula
    cv2.imwrite(output_path, binary)
    plt.imshow(image, cmap='gray')
    plt.title('Processed DICOM Image')
    plt.axis('off')
    plt.show()  
print('амжилттай')

