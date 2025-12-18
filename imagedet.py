import cv2
import numpy as np
from matplotlib import pyplot as plt

def create_lumbar_spine_xray():
    """L1-L2 нугалмын симуляци зураг үүсгэх"""
    img = np.zeros((600, 400), dtype=np.uint8)
    
    bone_color = 200
    
    # L1 нугалам (дээд)
    # Их бие (vertebral body)
    cv2.rectangle(img, (150, 150), (250, 210), bone_color, -1)
    # Spinous process (арын сунасан хэсэг)
    pts_l1_spine = np.array([[200, 150], [180, 120], [220, 120]], np.int32)
    cv2.fillPoly(img, [pts_l1_spine], bone_color)
    # Transverse process (хажуугийн сунасан хэсэг)
    cv2.ellipse(img, (140, 175), (15, 25), 0, 0, 360, bone_color, -1)
    cv2.ellipse(img, (260, 175), (15, 25), 0, 0, 360, bone_color, -1)
    
    # Диск (L1-L2 хоорондын)
    cv2.rectangle(img, (160, 210), (240, 230), bone_color-60, -1)
    
    # L2 нугалам (доод)
    # Их бие
    cv2.rectangle(img, (150, 230), (250, 290), bone_color, -1)
    # Spinous process
    pts_l2_spine = np.array([[200, 290], [180, 320], [220, 320]], np.int32)
    cv2.fillPoly(img, [pts_l2_spine], bone_color)
    # Transverse process
    cv2.ellipse(img, (140, 255), (15, 25), 0, 0, 360, bone_color, -1)
    cv2.ellipse(img, (260, 255), (15, 25), 0, 0, 360, bone_color, -1)
    
    # *** ХУГАРАЛ НЭМЭХ - L2 их биеийн дээд хэсэгт ***
    # Compression fracture (шахагдсан хугарал)
    # L2-ийн дээд талд зигзаг хугарал
    fracture_line = [
        (155, 235), (170, 238), (165, 242), (180, 245),
        (175, 248), (190, 250), (185, 253), (200, 255),
        (195, 258), (210, 260), (205, 263), (220, 265),
        (215, 268), (230, 270), (235, 273), (245, 275)
    ]
    
    for i in range(len(fracture_line) - 1):
        cv2.line(img, fracture_line[i], fracture_line[i+1], 20, 4)
    
    # Хугарлын эргэн тойронд бага зэрэг деформаци
    cv2.ellipse(img, (200, 250), (40, 15), 0, 0, 360, bone_color-80, -1)
    
    # Хугарлаас болж дээд хэсэг бага зэрэг шахагдсан
    cv2.line(img, (155, 235), (245, 235), bone_color-100, 3)
    
    # Дуу чимээ нэмэх (бодит рентген шиг)
    noise = np.random.normal(0, 10, img.shape).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    # Gradient (дэвсгэр тэгш бус гэрэлтэлт)
    for i in range(img.shape[0]):
        factor = 0.6 + 0.4 * np.sin(i / 100)
        img[i, :] = np.clip(img[i, :] * factor, 0, 255).astype(np.uint8)
    
    # Хөндлөн гэрлийн шугам (рентген эффект)
    cv2.line(img, (0, 300), (400, 320), 255, 2, cv2.LINE_AA)
    
    # Тэмдэглэгээ нэмэх
    cv2.putText(img, 'L1', (270, 180), cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, 255, 2, cv2.LINE_AA)
    cv2.putText(img, 'L2', (270, 260), cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, 255, 2, cv2.LINE_AA)
    
    return img

def detect_vertebral_fracture(img):
    """Нугалмын хугарал илрүүлэх"""
    
    # Preprocessing
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    
    # CLAHE - contrast сайжруулалт
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(blurred)
    
    # Threshold
    _, thresh = cv2.threshold(enhanced, 0, 255, 
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Инверс хийх (хугарал харанхуй учраас)
    thresh_inv = cv2.bitwise_not(thresh)
    
    # Morphological operations
    kernel = np.ones((3, 3), np.uint8)
    morph = cv2.morphologyEx(thresh_inv, cv2.MORPH_CLOSE, kernel, iterations=2)
    morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Canny edge detection
    edges = cv2.Canny(enhanced, 20, 80)
    
    # Contour илрүүлэх
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, 
                                          cv2.CHAIN_APPROX_SIMPLE)
    
    # Үр дүн зураг
    result = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    # Хугарал илрүүлэх - L1-L2 бүсэд анхаарах
    fractures = []
    
    # L1-L2 бүс (y = 150-300 орчим)
    roi_y_min, roi_y_max = 150, 300
    
    for contour in contours:
        area = cv2.contourArea(contour)
        
        if area < 30 or area > 3000:
            continue
        
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            continue
        
        # Bounding box
        x, y, w, h = cv2.boundingRect(contour)
        
        # L1-L2 бүс дотор байгаа эсэхийг шалгах
        if not (roi_y_min < y < roi_y_max):
            continue
        
        # Circularity (дугуй байдал)
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        
        # Aspect ratio
        aspect_ratio = float(w) / h if h > 0 else 0
        
        # Хугарал шалгуур:
        # 1. Сунасан хэлбэртэй (circularity < 0.3)
        # 2. Хэвтээ чиглэлтэй (aspect_ratio > 2)
        # 3. Тодорхой хэмжээтэй
        
        if circularity < 0.35 and aspect_ratio > 1.5 and area > 50:
            fractures.append({
                'contour': contour,
                'area': area,
                'circularity': circularity,
                'position': (x, y),
                'aspect_ratio': aspect_ratio
            })
            
            # Хугарлыг тэмдэглэх
            cv2.drawContours(result, [contour], -1, (0, 0, 255), 3)
            cv2.rectangle(result, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # "FRACTURE" гэж бичих
            cv2.putText(result, 'FRACTURE!', (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    # ROI хүрээг харуулах
    cv2.rectangle(result, (0, roi_y_min), (img.shape[1], roi_y_max), 
                 (255, 255, 0), 2)
    cv2.putText(result, 'L1-L2 Region', (10, roi_y_min-10), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    
    return result, thresh, edges, morph, fractures

def main():
    print("="*60)
    print("  L1-L2 НУГАЛМЫН ХУГАРАЛ ИЛРҮҮЛЭХ СИСТЕМ")
    print("="*60)
    
    print("\n[1/3] Нурууны нугалам симуляци үүсгэж байна...")
    xray = create_lumbar_spine_xray()
    
    print("[2/3] Хугарал илрүүлж байна...")
    result, thresh, edges, morph, fractures = detect_vertebral_fracture(xray)
    
    print("[3/3] Үр дүн боловсруулж байна...")
    
    # Visualization
    plt.figure(figsize=(18, 12))
    
    # Эх зураг
    plt.subplot(2, 4, 1)
    plt.imshow(xray, cmap='gray')
    plt.title('Эх зураг\n(L1-L2 нугалам)', fontsize=12, weight='bold')
    plt.axis('off')
    
    # Enhanced
    plt.subplot(2, 4, 2)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(cv2.GaussianBlur(xray, (5, 5), 0))
    plt.imshow(enhanced, cmap='gray')
    plt.title('CLAHE Enhanced', fontsize=11)
    plt.axis('off')
    
    # Threshold
    plt.subplot(2, 4, 3)
    plt.imshow(thresh, cmap='gray')
    plt.title('Threshold', fontsize=11)
    plt.axis('off')
    
    # Morphological
    plt.subplot(2, 4, 4)
    plt.imshow(morph, cmap='gray')
    plt.title('Morphological', fontsize=11)
    plt.axis('off')
    
    # Edges
    plt.subplot(2, 4, 5)
    plt.imshow(edges, cmap='gray')
    plt.title('Canny Edges', fontsize=11)
    plt.axis('off')
    
    # Илрүүлсэн үр дүн
    plt.subplot(2, 4, 6)
    plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    title_color = 'red' if len(fractures) > 0 else 'green'
    plt.title(f'ИЛРҮҮЛЭЛТ\n({len(fractures)} хугарал)', 
             fontsize=13, color=title_color, weight='bold')
    plt.axis('off')
    
    # Томруулсан харагдац
    plt.subplot(2, 4, 7)
    if len(fractures) > 0:
        x, y = fractures[0]['position']
        margin = 50
        y1 = max(0, y - margin)
        y2 = min(result.shape[0], y + margin + 50)
        x1 = max(0, x - margin)
        x2 = min(result.shape[1], x + margin + 100)
        zoomed = result[y1:y2, x1:x2]
        plt.imshow(cv2.cvtColor(zoomed, cv2.COLOR_BGR2RGB))
        plt.title('Хугарлын томруулсан', fontsize=11, color='red')
    else:
        plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        plt.title('Хугарал олдсонгүй', fontsize=11, color='green')
    plt.axis('off')
    
    # Статистик мэдээлэл
    plt.subplot(2, 4, 8)
    plt.axis('off')
    
    stats = "╔═══════════════════════════╗\n"
    stats += "║   ШИНЖИЛГЭЭНИЙ ҮР ДҮН    ║\n"
    stats += "╚═══════════════════════════╝\n\n"
    stats += f"Илэрсэн хугарал: {len(fractures)}\n"
    stats += f"Шалгасан бүс: L1-L2\n\n"
    
    if fractures:
        stats += "⚠️  АНХААРУУЛГА!\n"
        stats += "Хугарал илэрлээ!\n\n"
        stats += "Дэлгэрэнгүй:\n"
        for i, f in enumerate(fractures[:2], 1):
            stats += f"\nХугарал #{i}:\n"
            stats += f"  Талбай: {f['area']:.0f} px²\n"
            stats += f"  Дугуй: {f['circularity']:.3f}\n"
            stats += f"  Харьцаа: {f['aspect_ratio']:.2f}\n"
        stats += "\n→ Эмчид шалгуулна уу!"
    else:
        stats += "✓ Хугарал илрээгүй\n"
        stats += "✓ Нугалам хэвийн байна"
    
    plt.text(0.05, 0.5, stats, fontsize=10, family='monospace',
             verticalalignment='center',
             bbox=dict(boxstyle='round', facecolor='lightyellow', 
                      edgecolor='black', linewidth=2, alpha=0.9))
    
    plt.tight_layout()
    plt.show()
    
    # Terminal дээр мэдээлэл
    print("\n" + "="*60)
    if fractures:
        print("⚠️  ХУГАРАЛ ИЛЭРЛЭЭ!")
        print("="*60)
        print(f"Нийт илэрсэн: {len(fractures)}")
        for i, f in enumerate(fractures, 1):
            print(f"\nХугарал #{i}:")
            print(f"  Байршил: L1-L2 бүс")
            print(f"  Талбай: {f['area']:.1f} пиксел²")
            print(f"  Өргөн/Өндөр: {f['aspect_ratio']:.2f}")
    else:
        print("✓ ХУГАРАЛ ИЛРЭЭГҮЙ")
        print("="*60)
        print("Нугалам хэвийн байна.")
    
    print("\n📌 Тэмдэглэл:")
    print("   - Энэ нь демо програм")
    print("   - Эмнэлгийн оношлогоо биш")
    print("   - Эмчид үзүүлэхийг зөвлөж байна")
    print("="*60)

if __name__ == "__main__":
    main()