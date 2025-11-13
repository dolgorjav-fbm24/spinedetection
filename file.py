import pydicom as dicom
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

def load_dicom_image(file_path):
    """Load a DICOM image and return the pixel array."""
    dicom_data = dicom.dcmread(file_path)
    return dicom_data.pixel_array

path="./img/example1.dcm"
x=dicom.dcmread(path)
print(dir(x))
