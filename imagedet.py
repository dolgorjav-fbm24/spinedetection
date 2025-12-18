import cv2
import numpy as np
import pydicom as dicom
import matplotlib.pyplot as plt
img=cv2.imread("./img/sample2.dcm",cv2.IMREAD_GRAYSCALE)

def preprocess_dicom_image(dicom_path):
    ds = dicom.dcmread(dicom_path)
    img = ds.pixel_array   
    #DICOM зургийг уншиж, боловсруулна  
    print()
    #сааралт зураг болгон хувиргах
    if len(img.shape)==3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else: gray=img.copy() 
    gray = gray.astype(np.uint8)
    denoised = cv2.fastNlMeansDenoising(gray, None, h=10, 
                                        templateWindowSize=7,
                                        searchWindowSize=21)
                                    # h=3-20 дуу чимээг арилгах хүч, тохиромжтой утга нь 10
                                    # templateWindowSize=7  загвар цонхны хэмжээ (ихэвчлэн 7)
                                    # searchWindowSize=21 хайх цонхны хэмжээ (ихэвчлэн
    print(denoised.shape)
    # Clahe-контраст сайжруулах
    clahe=cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    enhanced=clahe.apply(denoised)
    # NormalizatiON  hiih
    normalized=cv2. normalize(enhanced, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX )
   
    gaussian = cv2.GaussianBlur(normalized, (0, 0), 1.0)
    sharpened = cv2.addWeighted(normalized, 1.3, gaussian, -0.3, 0)
    #jijig duu chimeeg arilgana-morphological operation
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))   
    cleaned = cv2.morphologyEx(sharpened, cv2.MORPH_OPEN, kernel)
    
    print(  )
    return cleaned, gray
processed_img, original_img = preprocess_dicom_image("./img/sample2.dcm")
# Үр дүнг харуулах
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title('Original DICOM Image')

plt.imshow(original_img, cmap='gray')
plt.axis('off')
plt.subplot(1, 2, 2)
plt.title('Processed Image')
plt.imshow(processed_img, cmap='gray')
plt.axis('off')
plt.show()
"DICOM зургийг амжилттай боловсрууллаа."
print("Амжилттай DICOM зургийг боловсрууллаа.") 


