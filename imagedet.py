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

def enhance_vertebrae(img):
    """Нугалмыг тодруулах (зөвхөн CLAHE)"""
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    enhanced = clahe.apply(img)
    return enhanced

def detect_vertebrae_regions(img):
    """Нугалмын бүс олох - илүү зөөлөн арга"""
    # 1. Median blur - дуу чимээ арилгах
    blurred = cv2.medianBlur(img, 5)
    
    # 2. Adaptive threshold - орон нутгийн босго
    binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY, 21, 5)
    
    # 3. Том morphological operations - жижиг дуу чимээ устгах
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=3)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
    
    # 4. Контурууд олох
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    return contours, binary

def filter_and_identify_vertebrae(contours, img_shape):
    """L1-L5 нугалмыг шүүж таних"""
    height, width = img_shape
    candidates = []
    
    print(f"\n   Зургийн хэмжээ: {width}x{height}, Нийт талбай: {width*height}")
    
    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        x, y, w, h = cv2.boundingRect(contour)
        
        # Хэмжээний шалгуур - илүү том объект хайх
        min_area = (height * width) * 0.01   # 1% - илүү том
        max_area = (height * width) * 0.25   # 25%
        
        # Aspect ratio - нугалам нь ойролцоогоор дөрвөлжин
        aspect_ratio = float(w) / h if h > 0 else 0
        
        # Дунд хэсэгт байгаа эсэх
        center_x = x + w // 2
        is_centered = 0.15 * width < center_x < 0.85 * width
        
        # Debug: эхний 15 контурын мэдээлэл
        if i < 15:
            print(f"   Контур {i+1}: area={area:.0f} ({'✓' if min_area < area < max_area else '✗'}), "
                  f"size={w}×{h}, ratio={aspect_ratio:.2f} ({'✓' if 0.3 < aspect_ratio < 3.0 else '✗'}), "
                  f"centered={'✓' if is_centered else '✗'}")
        
        if (min_area < area < max_area and 
            0.3 < aspect_ratio < 3.0 and
            w > 30 and h > 30 and
            is_centered):
            
            candidates.append({
                'contour': contour,
                'bbox': (x, y, w, h),
                'area': area,
                'center_y': y + h//2,
                'center_x': x + w//2
            })
    
    print(f"\n   → {len(candidates)} candidates шүүгдсэн")
    
    # Y координатаар эрэмбэлэх (дээрээс доош)
    candidates = sorted(candidates, key=lambda x: x['center_y'])
    
    # Хамгийн том 5-10 контур авах, дараа нь Y-ээр эрэмбэлэх
    if len(candidates) > 5:
        # Area-гаар эрэмбэлж том 8-ыг авах
        candidates_by_size = sorted(candidates, key=lambda x: x['area'], reverse=True)[:8]
        # Дахиад Y-ээр эрэмбэлэх
        candidates_by_size = sorted(candidates_by_size, key=lambda x: x['center_y'])
        vertebrae = candidates_by_size[:5]
    else:
        vertebrae = candidates[:5]
    
    # L1-L5 label
    labels = ['L1', 'L2', 'L3', 'L4', 'L5']
    for i, vert in enumerate(vertebrae):
        vert['label'] = labels[i] if i < 5 else f'V{i+1}'
    
    return vertebrae

def draw_vertebrae_boxes(img, vertebrae):
    """Анхны зураг дээр дөрвөлжин зурах"""
    # RGB болгох
    img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    # Өнгөний палитр
    colors = [
        (0, 255, 255),    # Шар - L1
        (0, 255, 0),      # Ногоон - L2
        (255, 0, 0),      # Цэнхэр - L3
        (0, 165, 255),    # Улбар шар - L4
        (255, 0, 255)     # Ягаан - L5
    ]
    
    for i, vert in enumerate(vertebrae):
        x, y, w, h = vert['bbox']
        color = colors[i % len(colors)]
        label = vert['label']
        
        # Том дөрвөлжин (зузаан 4)
        cv2.rectangle(img_color, (x, y), (x+w, y+h), color, 4)
        
        # Label бичих
        font_scale = 1.2
        thickness = 3
        
        # Background box
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_BOLD, font_scale, thickness)[0]
        cv2.rectangle(img_color, 
                     (x-2, y-text_size[1]-15), 
                     (x+text_size[0]+8, y-2), 
                     color, -1)
        
        # Text (цагаан өнгөөр)
        cv2.putText(img_color, label, (x+3, y-8), 
                   cv2.FONT_HERSHEY_BOLD, font_scale, (255, 255, 255), thickness)
        
        # Төв цэг
        center = (vert['center_x'], vert['center_y'])
        cv2.circle(img_color, center, 6, color, -1)
        cv2.circle(img_color, center, 6, (255, 255, 255), 2)
    
    return img_color

def create_result_display(original, result_img, vertebrae):
    """Үр дүн харуулах - анхны болон тэмдэглэсэн зургийг зэрэгцүүлэх"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 10))
    
    # 1. Анхны зураг
    axes[0].imshow(original, cmap='gray')
    axes[0].set_title('Анхны DICOM зураг', fontsize=16, fontweight='bold', pad=20)
    axes[0].axis('off')
    
    # 2. Үр дүн
    axes[1].imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
    title = f'L1-L5 нугалмын илрүүлэлт ({len(vertebrae)} олдсон)'
    axes[1].set_title(title, fontsize=16, fontweight='bold', pad=20)
    axes[1].axis('off')
    
    plt.tight_layout()
    
    # Мэдээлэл хэвлэх
    print("\n" + "="*60)
    print("ОЛДСОН НУГАЛМУУД:")
    print("="*60)
    for vert in vertebrae:
        x, y, w, h = vert['bbox']
        print(f"{vert['label']:3s}: Байршил=({x:4d},{y:4d}), "
              f"Хэмжээ={w:3d}×{h:3d}px, Талбай={vert['area']:8.0f}px²")
    print("="*60)
    
    plt.show()

# ==================== MAIN PROGRAM ====================

print("\n" + "="*60)
print("L1-L5 НУГАЛМЫН ИЛРҮҮЛЭЛТ")
print("="*60 + "\n")

# 1. DICOM зураг унших
dicom_path = "img/example1.dcm"
original_img = load_dicom_image(dicom_path)

if original_img is not None:
    # 2. Бага зэрэг сайжруулах (зөвхөн CLAHE)
    print("[1/4] Зургийг бага зэрэг сайжруулж байна...")
    enhanced = enhance_vertebrae(original_img)
    
    # 3. Нугалмын бүс олох
    print("[2/4] Нугалмын бүс олж байна...")
    contours, binary = detect_vertebrae_regions(enhanced)
    print(f"      → {len(contours)} контур олдсон")
    
    # 4. L1-L5 шүүж таних
    print("[3/4] L1-L5 таниж байна...")
    vertebrae = filter_and_identify_vertebrae(contours, original_img.shape)
    print(f"      → {len(vertebrae)} нугалам таньсан")
    
    if len(vertebrae) > 0:
        # 5. Анхны зураг дээр дөрвөлжин зурах
        print("[4/4] Үр дүн зурж байна...")
        result_img = draw_vertebrae_boxes(original_img, vertebrae)
        
        # 6. Харуулах
        create_result_display(original_img, result_img, vertebrae)
        
        print("\n✅ АМЖИЛТТАЙ ДУУСЛАА!\n")
    else:
        print("\n⚠️  Нугалам олдсонгүй. Параметрүүдийг тохируулах хэрэгтэй.")
        
        # Debug: бүх контурын мэдээлэл харуулах
        print("\nБүх контурын мэдээлэл:")
        for i, c in enumerate(contours[:10]):
            area = cv2.contourArea(c)
            x, y, w, h = cv2.boundingRect(c)
            print(f"Контур {i+1}: area={area:.0f}, size={w}x{h}, pos=({x},{y})")
        
else:
    print(f"\n❌ Зураг олдсонгүй: {dicom_path}\n")