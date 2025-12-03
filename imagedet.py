import cv2
import numpy as np
import pydicom as dicom
import matplotlib.pyplot as plt
img=cv2.imread("./img/sample2.dcm",cv2.IMREAD_GRAYSCALE)

def preprocess_dicom_image(file_path):
    #DICOM зургийг уншиж, боловсруулна
    print()
    #сааралт зураг болгон хувиргах
    if len(img.shape)==3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else: gray=img.copy()   
    denoised = cv2.fastNlMeansDenoising(gray, None, h=10, 
                                        templateWindowSize=7,
                                        searchWindowSize=21)
                                    # h=3-20 дуу чимээг арилгах хүч, тохиромжтой утга нь 10
                                    # templateWindowSize=7  загвар цонхны хэмжээ (ихэвчлэн 7)
                                    # searchWindowSize=21 хайх цонхны хэмжээ (ихэвчлэн
