import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def load_image(path):
    """Зураг уншина (DICOM эсвэл энгийн зураг)"""
    try:
        # DICOM эсэхийг шалгах
        if path.lower().endswith('.dcm'):
            import pydicom as dicom
            ds = dicom.dcmread(path)
            img = ds.pixel_array
            # Normalize to 8-bit
             
        else:
            # Энгийн зураг (PNG, JPG)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            
        if img is None:
            raise ValueError("Зураг уншигдсангүй")
            
        print(f"✅ Зураг уншигдлаа: {img.shape}")
        return img
        
    except FileNotFoundError:
        print(f"❌ Файл олдсонгүй: {path}")
        print(f"💡 Одоогийн зам: {os.getcwd()}")
        return None
    except Exception as e:
        print(f"❌ Алдаа: {e}")
        return None

def preprocess_spine_image(img):
    """Нурууны зургийг боловсруулна"""
    # 1. Contrast enhancement (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    enhanced = clahe.apply(img)
    
    # 2. Denoising
    denoised = cv2.fastNlMeansDenoising(enhanced, None, h=10, 
                                        templateWindowSize=7, 
                                        searchWindowSize=21)
    
    # 3. Histogram equalization
    equalized = cv2.equalizeHist(denoised)
    
    return enhanced, denoised, equalized

def detect_spine_region(img):
    """Нурууны бүс олно"""
    # Edge detection
    edges = cv2.Canny(img, 50, 150)
    
    # Морфологи - edges-ийг холбох
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 15))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    
    # Contours олох
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, 
                                   cv2.CHAIN_APPROX_SIMPLE)
    
    # Хамгийн урт contour (нурууны мөр)
    if contours:
        spine_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(spine_contour)
        return edges, closed, (x, y, w, h), spine_contour
    
    return edges, closed, None, None

def detect_vertebrae(img, spine_region):
    """L1-L5 нугалмуудыг олно"""
    if spine_region is None:
        print("⚠️ Нурууны бүс олдсонгүй")
        return [], img
    
    x, y, w, h = spine_region
    
    # Нурууны бүсийг crop хийх
    roi = img[y:y+h, x:x+w]
    
    # Adaptive threshold
    binary = cv2.adaptiveThreshold(roi, 255, 
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 15, 3)
    
    # Морфологи - нугалмын хэлбэрийг сайжруулах
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    morph = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Contours олох
    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, 
                                   cv2.CHAIN_APPROX_SIMPLE)
    
    # Нугалмуудыг шүүх
    vertebrae = []
    min_area = (w * h) * 0.01  # ROI-ийн 1%-иас их
    max_area = (w * h) * 0.15  # ROI-ийн 15%-иас бага
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if min_area < area < max_area:
            # Bounding box
            bx, by, bw, bh = cv2.boundingRect(cnt)
            
            # Aspect ratio шалгах (нугалмын хэлбэр)
            aspect_ratio = bw / float(bh) if bh > 0 else 0
            
            if 0.5 < aspect_ratio < 2.5:
                # Глобал координат руу хөрвүүлэх
                global_x = x + bx
                global_y = y + by
                vertebrae.append({
                    'bbox': (global_x, global_y, bw, bh),
                    'area': area,
                    'contour': cnt,
                    'center_y': global_y + bh//2
                })
    
    # Y координатаар эрэмбэлэх (дээрээс доош: L1->L5)
    vertebrae.sort(key=lambda v: v['center_y'])
    
    return vertebrae[:5], morph  # Зөвхөн эхний 5-ыг авах (L1-L5)

def visualize_results(original, enhanced, edges, spine_region, vertebrae, morph):
    """Үр дүнг харуулна"""
    # Үр дүн зураг үүсгэх
    result = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
    
    # Нурууны бүс зурах
    if spine_region:
        x, y, w, h = spine_region
        cv2.rectangle(result, (x, y), (x+w, y+h), (255, 0, 0), 2)
    
    # Нугалмуудыг зурах
    labels = ['L1', 'L2', 'L3', 'L4', 'L5']
    colors = [(0, 255, 0), (0, 255, 255), (255, 255, 0), 
              (255, 128, 0), (255, 0, 255)]
    
    for i, vert in enumerate(vertebrae):
        bbox = vert['bbox']
        x, y, w, h = bbox
        
        # Bounding box
        color = colors[i] if i < len(colors) else (0, 255, 0)
        cv2.rectangle(result, (x, y), (x+w, y+h), color, 2)
        
        # Label
        label = labels[i] if i < len(labels) else f"V{i+1}"
        cv2.putText(result, label, (x-30, y+h//2), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    
    # Visualization
    fig = plt.figure(figsize=(16, 10))
    
    # Layout
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(original, cmap='gray')
    ax1.set_title('1. Анхны зураг', fontsize=12, fontweight='bold')
    ax1.axis('off')
    
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(enhanced, cmap='gray')
    ax2.set_title('2. Enhanced (CLAHE)', fontsize=12, fontweight='bold')
    ax2.axis('off')
    
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(edges, cmap='gray')
    ax3.set_title('3. Edge Detection', fontsize=12, fontweight='bold')
    ax3.axis('off')
    
    if morph is not None:
        ax4 = fig.add_subplot(gs[1, 0])
        ax4.imshow(morph, cmap='gray')
        ax4.set_title('4. Morphology (ROI)', fontsize=12, fontweight='bold')
        ax4.axis('off')
    
    ax5 = fig.add_subplot(gs[1, 1:])
    ax5.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    ax5.set_title(f'5. Үр дүн: {len(vertebrae)} нугалам олдлоо', 
                 fontsize=14, fontweight='bold', color='green')
    ax5.axis('off')
    
    plt.suptitle('L1-L5 Нугалам Detection - OpenCV', 
                fontsize=16, fontweight='bold', y=0.98)
    
    plt.show()
    
    # Статистик хэвлэх
    print("\n" + "="*50)
    print("📊 DETECTION ҮР ДҮН")
    print("="*50)
    for i, vert in enumerate(vertebrae):
        label = labels[i] if i < len(labels) else f"V{i+1}"
        bbox = vert['bbox']
        area = vert['area']
        print(f"{label}: bbox={bbox}, area={area:.0f} px²")
    print("="*50)

def main():
    """Main функц"""
    print("🦴 L1-L5 НУГАЛАМ DETECTION - OpenCV")
    print("="*50)
    
    # Файлын зам (өөрчилж болно)
    image_path = './img/example1.dcm'  # Танай файл
    
    # 1. Зураг уншина
    print("\n📂 Зураг уншиж байна...")
    img = load_image(image_path)
    
    if img is None:
        print("\n💡 Зөвлөгөө:")
        print("1. VinDr-SpineXR dataset татаж аваарай")
        print("2. Зургийг './img/' folder дотор хадгалаарай")
        print("3. Кодон дахь 'image_path' өөрчилнө үү")
        return
    
    # 2. Preprocessing
    print("\n🔧 Зураг боловсруулж байна...")
    enhanced, denoised, equalized = preprocess_spine_image(img)
    
    # 3. Нурууны бүс олох
    print("\n🔍 Нурууны бүс хайж байна...")
    edges, closed, spine_region, spine_contour = detect_spine_region(equalized)
    
    if spine_region:
        x, y, w, h = spine_region
        print(f"✅ Нурууны бүс олдлоо: ({x}, {y}, {w}, {h})")
    
    # 4. Нугалмуудыг олох
    print("\n🦴 L1-L5 нугалмуудыг хайж байна...")
    vertebrae, morph = detect_vertebrae(equalized, spine_region)
    
    print(f"✅ {len(vertebrae)} нугалам олдлоо!")
    
    # 5. Үр дүн харуулах
    print("\n📊 Үр дүн харуулж байна...")
    visualize_results(img, enhanced, edges, spine_region, vertebrae, morph)
    
    print("\n✅ Дууслаа!")
    print("\n💡 Дараагийн алхам: YOLO model сургах")

if __name__ == "__main__":
    main()
    # зөвхөн жишээ файл тул мөр бүрийг судлах.

    