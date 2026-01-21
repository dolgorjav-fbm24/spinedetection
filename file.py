import cv2
import numpy as np
import pydicom
from matplotlib import pyplot as plt
import os

def load_dicom_image(dicom_path):
    """DICOM файл уншиж numpy array болгох"""
    try:
        dicom = pydicom.dcmread(dicom_path)
        img_array = dicom.pixel_array
        
        # Normalize (0-255)
        img_array = img_array.astype(float)
        img_normalized = ((img_array - img_array.min()) / 
                         (img_array.max() - img_array.min()) * 255)
        img_normalized = img_normalized.astype(np.uint8)
        
        print(f"✓ Зураг уншсан: {img_normalized.shape}")
        return img_normalized
    except Exception as e:
        print(f"❌ Алдаа: {e}")
        return None

def preprocess_image(img):
    """Зургийг preprocessing хийх"""
    # 1. Gaussian blur - дуу чимээ арилгах
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    
    # 2. CLAHE - contrast сайжруулах
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(blurred)
    
    return enhanced

def detect_edges(img):
    """Ирмэг олох (Edge Detection)"""
    # Canny edge detection
    edges = cv2.Canny(img, 50, 150)
    return edges

def find_contours(img):
    """Контур олох"""
    # Threshold хийх
    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Контурууд олох
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Контуруудыг хэмжээгээр эрэмбэлэх
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    return contours, binary

def draw_contours(img, contours, top_n=5):
    """Контурууд зурах"""
    # RGB болгох (өнгөтэй зурахын тулд)
    img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    # Хамгийн том contour-ууд зурах
    for i, contour in enumerate(contours[:top_n]):
        # Өнгө сонгох
        color = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)][i % 5]
        
        # Контур зурах
        cv2.drawContours(img_color, [contour], -1, color, 2)
        
        # Талбай бичих
        area = cv2.contourArea(contour)
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            cv2.putText(img_color, f"#{i+1}: {area:.0f}px", (cx-40, cy), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    return img_color

def detect_shapes(img, contours, top_n=5):
    """Дүрс тодорхойлох (гурвалжин, дөрвөлжин, тойрог гэх мэт)"""
    results = []
    
    for i, contour in enumerate(contours[:top_n]):
        # Contour-ын мэдээлэл
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)
        
        # Дүрс тодорхойлох
        vertices = len(approx)
        if vertices == 3:
            shape = "Гурвалжин"
        elif vertices == 4:
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = float(w) / h
            shape = "Квадрат" if 0.95 <= aspect_ratio <= 1.05 else "Тэгш өнцөгт"
        elif vertices > 4:
            shape = "Тойрог/Эллипс"
        else:
            shape = "Тодорхойгүй"
        
        results.append({
            'index': i+1,
            'shape': shape,
            'area': area,
            'perimeter': perimeter,
            'vertices': vertices
        })
    
    return results

def visualize_all(original, preprocessed, edges, binary, contour_img):
    """Бүх үр дүнг харуулах"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    axes[0, 0].imshow(original, cmap='gray')
    axes[0, 0].set_title('1. Анхны зураг')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(preprocessed, cmap='gray')
    axes[0, 1].set_title('2. Preprocessing')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(edges, cmap='gray')
    axes[0, 2].set_title('3. Edge Detection (Canny)')
    axes[0, 2].axis('off')
    
    axes[1, 0].imshow(binary, cmap='gray')
    axes[1, 0].set_title('4. Binary Threshold')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(cv2.cvtColor(contour_img, cv2.COLOR_BGR2RGB))
    axes[1, 1].set_title('5. Контурууд')
    axes[1, 1].axis('off')
    
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.show()

# ==================== MAIN PROGRAM ====================

# 1. DICOM зураг уншиx
dicom_path = "img/example1.dcm"
img = load_dicom_image(dicom_path)

if img is not None:
    print("\n" + "="*50)
    print("IMAGE PROCESSING ЭХЭЛЛЭЭ")
    print("="*50)
    
    # 2. Preprocessing
    preprocessed = preprocess_image(img)
    print("✓ Preprocessing хийгдсэн")
    
    # 3. Edge Detection
    edges = detect_edges(preprocessed)
    print("✓ Edge detection хийгдсэн")
    
    # 4. Contour Detection
    contours, binary = find_contours(preprocessed)
    print(f"✓ {len(contours)} контур олдсон")
    
    # 5. Контур зурах
    contour_img = draw_contours(preprocessed, contours, top_n=5)
    print("✓ Контурууд зурагдсан")
    
    # 6. Дүрс тодорхойлох
    shapes = detect_shapes(preprocessed, contours, top_n=5)
    print("\n" + "="*50)
    print("ТОП 5 ОБЪЕКТ:")
    print("="*50)
    for shape in shapes:
        print(f"#{shape['index']}: {shape['shape']} - Талбай: {shape['area']:.0f}px, "
              f"Периметр: {shape['perimeter']:.0f}px, Оройнууд: {shape['vertices']}")
    
    # 7. Бүх үр дүнг харуулах
    print("\n✓ Үр дүн харуулж байна...")
    visualize_all(img, preprocessed, edges, binary, contour_img)
    
    print("\n✅ ДУУСЛАА!")
else:
    print(f"❌ Зураг олдсонгүй: {dicom_path}")
    