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

def preprocess_for_vertebrae(img):
    """Нугалам олоход тохирсон preprocessing"""
    # 1. CLAHE - яс сайн харагдуулах
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    enhanced = clahe.apply(img)
    
    # 2. Gaussian blur
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
    
    # 3. Morphological operations - структур сайжруулах
    kernel = np.ones((3,3), np.uint8)
    morph = cv2.morphologyEx(blurred, cv2.MORPH_CLOSE, kernel)
    
    return morph

def detect_vertebrae(img):
    """Нугалмыг олох"""
    # Adaptive threshold - background-оос ялгах
    binary = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 11, 2)
    
    # Morphological cleanup
    kernel = np.ones((5,5), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    
    # Контурууд олох
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    return contours, binary

def filter_vertebrae_contours(contours, img_shape):
    """Нугалам мэт харагдах контуруудыг шүүх"""
    height, width = img_shape
    candidates = []
    
    for contour in contours:
        area = cv2.contourArea(contour)
        x, y, w, h = cv2.boundingRect(contour)
        
        # Хэмжээний шалгуур
        min_area = (height * width) * 0.005  # 0.5% -с их
        max_area = (height * width) * 0.15   # 15% -с бага
        
        # Aspect ratio шалгуур (нугалам нь тэгш өнцөгт төстэй)
        aspect_ratio = float(w) / h if h > 0 else 0
        
        # Шалгуурууд
        if (min_area < area < max_area and 
            0.4 < aspect_ratio < 2.5 and  # Өргөн ба өндрийн харьцаа
            w > 20 and h > 20):  # Хэтэрхий жижиг биш
            
            candidates.append({
                'contour': contour,
                'bbox': (x, y, w, h),
                'area': area,
                'center_y': y + h//2,
                'center_x': x + w//2
            })
    
    return candidates

def identify_l1_l5(candidates, img_shape):
    """L1-L5 нугалмыг таних (дээрээс доош эрэмбэлэх)"""
    height = img_shape[0]
    
    # Y координатаар эрэмбэлэх (дээрээс доош)
    candidates = sorted(candidates, key=lambda x: x['center_y'])
    
    # Дунд хэсэгт байгаа нугалмуудыг сонгох (ихэвчлэн дунд хэсэгт байдаг)
    middle_candidates = [c for c in candidates if 0.2*height < c['center_y'] < 0.8*height]
    
    # Хамгийн ойролцоо хэмжээтэй 5 нугалам сонгох
    if len(middle_candidates) >= 5:
        # Area-гаар ойролцоо байгаа 5-ыг авах
        middle_candidates = sorted(middle_candidates, key=lambda x: x['area'], reverse=True)
        vertebrae = middle_candidates[:5]
        vertebrae = sorted(vertebrae, key=lambda x: x['center_y'])
    else:
        vertebrae = middle_candidates
    
    # L1-L5 гэж label хийх
    labels = ['L1', 'L2', 'L3', 'L4', 'L5']
    for i, vert in enumerate(vertebrae[:5]):
        vert['label'] = labels[i] if i < len(labels) else f'L{i+1}'
    
    return vertebrae

def draw_vertebrae_boxes(img, vertebrae):
    """Нугалам дээр дөрвөлжин болон label зурах"""
    # RGB болгох
    img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    # Өнгөний палитр
    colors = [
        (255, 0, 0),    # Улаан - L1
        (0, 255, 0),    # Ногоон - L2
        (0, 0, 255),    # Цэнхэр - L3
        (255, 255, 0),  # Шар - L4
        (255, 0, 255)   # Ягаан - L5
    ]
    
    for i, vert in enumerate(vertebrae):
        x, y, w, h = vert['bbox']
        color = colors[i % len(colors)]
        label = vert['label']
        
        # Дөрвөлжин зурах (зузаан)
        cv2.rectangle(img_color, (x, y), (x+w, y+h), color, 3)
        
        # Label зурах (том фонт)
        font_scale = 1.5
        thickness = 3
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_BOLD, font_scale, thickness)[0]
        
        # Background rectangle
        cv2.rectangle(img_color, (x-5, y-text_size[1]-10), 
                     (x+text_size[0]+5, y), color, -1)
        
        # Text
        cv2.putText(img_color, label, (x, y-5), 
                   cv2.FONT_HERSHEY_BOLD, font_scale, (255, 255, 255), thickness)
        
        # Төв цэг зурах
        center = (vert['center_x'], vert['center_y'])
        cv2.circle(img_color, center, 5, color, -1)
    
    return img_color

def visualize_detection(original, preprocessed, binary, result_img, vertebrae):
    """Үр дүнг харуулах"""
    fig = plt.figure(figsize=(18, 12))
    
    # 1. Анхны зураг
    plt.subplot(2, 3, 1)
    plt.imshow(original, cmap='gray')
    plt.title('1. Анхны DICOM зураг', fontsize=14, fontweight='bold')
    plt.axis('off')
    
    # 2. Preprocessing
    plt.subplot(2, 3, 2)
    plt.imshow(preprocessed, cmap='gray')
    plt.title('2. Сайжруулсан зураг', fontsize=14, fontweight='bold')
    plt.axis('off')
    
    # 3. Binary
    plt.subplot(2, 3, 3)
    plt.imshow(binary, cmap='gray')
    plt.title('3. Binary threshold', fontsize=14, fontweight='bold')
    plt.axis('off')
    
    # 4. Үр дүн (том)
    plt.subplot(2, 2, 3)
    plt.imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
    plt.title(f'4. L1-L5 нугалмын илрүүлэлт ({len(vertebrae)} олдсон)', 
             fontsize=16, fontweight='bold')
    plt.axis('off')
    
    # 5. Мэдээлэл
    ax = plt.subplot(2, 2, 4)
    ax.axis('off')
    
    info_text = "НУГАЛМЫН МЭДЭЭЛЭЛ:\n" + "="*40 + "\n\n"
    for i, vert in enumerate(vertebrae):
        x, y, w, h = vert['bbox']
        info_text += f"{vert['label']}:\n"
        f"  • Байршил: ({x}, {y})\n"
        info_text += f"  • Хэмжээ: {w}×{h} px\n"
        info_text += f"  • Талбай: {vert['area']:.0f} px²\n\n"
    
    ax.text(0.1, 0.9, info_text, fontsize=11, verticalalignment='top',
           family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.show()

# ==================== MAIN PROGRAM ====================

print("="*60)
print("L1-L5 НУГАЛМЫН ИЛРҮҮЛЭЛТ")
print("="*60)

# 1. DICOM зураг унших
dicom_path = "img/example1.dcm"
img = load_dicom_image(dicom_path)

if img is not None:
    # 2. Preprocessing
    print("\n[1/5] Зураг боловсруулж байна...")
    preprocessed = preprocess_for_vertebrae(img)
    
    # 3. Нугалам олох
    print("[2/5] Контурууд олж байна...")
    contours, binary = detect_vertebrae(preprocessed)
    print(f"      → {len(contours)} контур олдсон")
    
    # 4. Нугалам шүүх
    print("[3/5] Нугалам шүүж байна...")
    candidates = filter_vertebrae_contours(contours, img.shape)
    print(f"      → {len(candidates)} candidates олдсон")
    
    # 5. L1-L5 таних
    print("[4/5] L1-L5 таниж байна...")
    vertebrae = identify_l1_l5(candidates, img.shape)
    print(f"      → {len(vertebrae)} нугалам таньсан")
    
    # 6. Зурах
    print("[5/5] Үр дүн зурж байна...")
    result_img = draw_vertebrae_boxes(preprocessed, vertebrae)
    
    # 7. Харуулах
    print("\n" + "="*60)
    print("ОЛДСОН НУГАЛМУУД:")
    print("="*60)
    for vert in vertebrae:
        x, y, w, h = vert['bbox']
        print(f"{vert['label']}: Байршил=({x},{y}), Хэмжээ={w}×{h}px, Талбай={vert['area']:.0f}px²")
    
    visualize_detection(img, preprocessed, binary, result_img, vertebrae)
    
    print("\n✅ АМЖИЛТТАЙ ДУУСЛАА!")
    
else:
    print(f"❌ Зураг олдсонгүй: {dicom_path}")
    