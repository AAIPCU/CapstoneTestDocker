# FFT
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import numpy as np
import cv2
from pathlib import Path
current_dir = Path(__file__).parent
model_path = current_dir / "spoof_detector_thai_scratch.pth"

class TruncatedVGG16(nn.Module):
    def __init__(self):
        super(TruncatedVGG16, self).__init__()
        vgg = models.vgg16(weights=None)
        self.features = nn.Sequential(*list(vgg.features.children())[:17])
    def forward(self, x): return self.features(x)

class CrossAttentionFusion(nn.Module):
    def __init__(self, channels=256):
        super(CrossAttentionFusion, self).__init__()
        self.mha_rgb = nn.MultiheadAttention(embed_dim=channels, num_heads=16, batch_first=True)
        self.mha_fft = nn.MultiheadAttention(embed_dim=channels, num_heads=16, batch_first=True)
        self.norm_rgb = nn.LayerNorm(channels)
        self.norm_fft = nn.LayerNorm(channels)
    def forward(self, x_rgb, x_fft):
        b, c, h, w = x_rgb.shape
        flat_rgb = x_rgb.view(b, c, -1).permute(0, 2, 1)
        flat_fft = x_fft.view(b, c, -1).permute(0, 2, 1)
        attn_rgb, _ = self.mha_rgb(query=flat_rgb, key=flat_fft, value=flat_fft)
        attn_fft, _ = self.mha_fft(query=flat_fft, key=flat_rgb, value=flat_rgb)
        out_rgb = self.norm_rgb(flat_rgb + attn_rgb)
        out_fft = self.norm_fft(flat_fft + attn_fft)
        return torch.cat([out_rgb.mean(dim=1), out_fft.mean(dim=1)], dim=1)

class SpoofDetector(nn.Module):
    def __init__(self):
        super(SpoofDetector, self).__init__()
        self.backbone_rgb = TruncatedVGG16()
        self.backbone_fft = TruncatedVGG16()
        self.fusion = CrossAttentionFusion()
        self.classifier = nn.Sequential(
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(128, 1)
        )
    def forward(self, rgb, fft):
        f1 = self.backbone_rgb(rgb)
        f2 = self.backbone_fft(fft)
        return self.classifier(self.fusion(f1, f2))

class FFTHighPassTransform:
    def __init__(self, radius=8, size=224):
        self.radius = radius
        self.size = size
    def __call__(self, img):
        img = img.resize((self.size, self.size), Image.Resampling.BICUBIC)
        img_gray = np.array(img.convert('L'))
        f = np.fft.fft2(img_gray)
        fshift = np.fft.fftshift(f)
        rows, cols = img_gray.shape
        crow, ccol = rows // 2, cols // 2
        y, x = np.ogrid[:rows, :cols]
        mask_area = (x - ccol)**2 + (y - crow)**2 >= self.radius**2
        mask = np.zeros((rows, cols), dtype=np.float32)
        mask[mask_area] = 1
        fshift_filtered = fshift * mask
        f_ishift = np.fft.ifftshift(fshift_filtered)
        img_back = np.fft.ifft2(f_ishift)
        img_back = np.abs(img_back)
        img_log = np.log(1 + img_back)
        m_min, m_max = np.min(img_log), np.max(img_log)
        img_normalized = (img_log - m_min) / (m_max - m_min + 1e-8)
        return torch.from_numpy(img_normalized).float().unsqueeze(0).repeat(3, 1, 1)

def preprocess_from_array(img_array):
    # 1. Convert BGR (OpenCV) to RGB (PIL)
    img_rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(img_rgb)

    # 2. RGB Branch Transform
    t_rgb = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    rgb_tensor = t_rgb(image).unsqueeze(0)

    # 3. FFT Branch Transform
    t_fft = FFTHighPassTransform(radius=8, size=224)
    fft_tensor = t_fft(image).unsqueeze(0)

    return rgb_tensor, fft_tensor

def spoofCheck(cropped_img_array):
    model_path = current_dir / "spoof_detector_thai_scratch.pth"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load Model
    model = SpoofDetector()
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except FileNotFoundError:
        print(f"Error: Model file '{model_path}' not found.")
        return
    model.to(device)
    model.eval()

    # Preprocess for Neural Network
    rgb, fft = preprocess_from_array(cropped_img_array)
    rgb = rgb.to(device)
    fft = fft.to(device)
    
    # Predict
    print("Running spoof detection...")
    with torch.no_grad():
        output = model(rgb, fft)
        score = torch.sigmoid(output).item()
        
    prediction = True if score > 0.5 else False # True=Spoof, False=Genuine
    
    print("\n" + "="*30)
    print(f"Result:     {prediction}")
    print(f"Score:      {score:.4f} (0=Genuine, 1=Spoof)")
    print("="*30)
    
    return score, prediction