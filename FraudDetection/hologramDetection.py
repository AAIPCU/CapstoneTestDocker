from ultralytics import YOLO
import cv2
from pathlib import Path

current_dir = Path(__file__).parent
model_path = current_dir / 'hologramYOLO.pt'

# Hologram Detection
def detect_hologram(image):
    model = YOLO(model_path)
    
    # Inference
    results = model.predict(image, conf=0.5)
    
    # Plot results
    for r in results:
        # im_array = r.plot()
        # im_rgb = cv2.cvtColor(im_array, cv2.COLOR_BGR2RGB)

        # cv2.imshow("Hologram Detection", im_rgb)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        
        # plt.figure(figsize=(10, 10))
        # plt.imshow(im_rgb)
        # plt.axis('off')
        # plt.show()

        max_conf = 0
        
        for box in r.boxes:
            print(f"Detected class {int(box.cls)} with confidence {float(box.conf):.2f}")
            if float(box.conf) > max_conf:
                max_conf = float(box.conf)
    if len(results) > 0 and len(results[0].boxes) > 0:
        return max_conf, True
    else:
        return max_conf,False
