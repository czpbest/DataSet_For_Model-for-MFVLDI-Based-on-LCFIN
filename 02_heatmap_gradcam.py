import os
import random
import numpy as np
import cv2
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# ====================== 固定随机种子 ======================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
set_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ====================== 网络模块（与训练代码一致）=====================
class MLPMixerLayer(nn.Module):
    def __init__(self, dim, seq_len, mlp_ratio=(0.5, 4.0)):
        super().__init__()
        mlp_dim1 = int(dim * mlp_ratio[0])
        self.norm1 = nn.LayerNorm(dim)
        self.token_mix = nn.Sequential(
            nn.Linear(seq_len, seq_len), nn.GELU(), nn.Linear(seq_len, seq_len)
        )
        self.norm2 = nn.LayerNorm(dim)
        self.channel_mix = nn.Sequential(
            nn.Linear(dim, mlp_dim1), nn.GELU(), nn.Linear(mlp_dim1, dim)
        )
    def forward(self, x):
        y = self.norm1(x).transpose(-1, -2)
        y = self.token_mix(y).transpose(-1, -2)
        x = x + y
        y = self.norm2(x)
        y = self.channel_mix(y)
        return x + y

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction), nn.ReLU(),
            nn.Linear(channels // reduction, channels), nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class BaseEfficientNetModel(nn.Module):
    def __init__(self, num_classes=12):
        super().__init__()
        self.backbone = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.DEFAULT)
        self.in_features = self.backbone.classifier[1].in_features
    def forward(self, x):
        return torch.flatten(self.backbone.avgpool(self.backbone.features(x)), 1)

class ModelC(BaseEfficientNetModel):
    def __init__(self, num_classes=12):
        super().__init__(num_classes)
        self.se_block = SEBlock(self.in_features)
        self.proj = nn.Conv2d(self.in_features, 512, kernel_size=1)
        self.seq_len = 100
        self.mixer_layers = nn.ModuleList([
            MLPMixerLayer(dim=512, seq_len=self.seq_len) for _ in range(4)
        ])
        self.classifier = nn.Linear(512, num_classes)
    def forward(self, x):
        x = self.backbone.features(x)
        x = self.se_block(x)
        x = self.proj(x)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        for mixer in self.mixer_layers:
            x = mixer(x)
        x = x.mean(dim=1)
        return self.classifier(x)

# ====================== 配置 ======================
WEIGHT_PATH = r"./train_output/best_model.pth"
DATA_ROOT = r"/workspace/水果叶片"
IMG_SIZE = 300
NUM_CLASSES = 12
OUTPUT_DIR = r"./heatmap_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CLASS_NAMES = sorted([d for d in os.listdir(os.path.join(DATA_ROOT, "test"))
                      if os.path.isdir(os.path.join(DATA_ROOT, "test", d))])

SHORT_NAMES = {
    "Apple___Apple_scab": "AAS",
    "Apple___Black_rot": "ABR",
    "Apple___Cedar_apple_rust": "ACAR",
    "Apple___healthy": "AH",
    "Grape___Black_rot": "GBR",
    "Grape___Esca_(Black_Measles)": "GEBM",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": "GLBILS",
    "Grape___healthy": "GH",
    "Tomato_Bacterial_spot": "TBS",
    "Tomato_Early_blight": "TEB",
    "Tomato_Leaf_Mold": "TLM",
    "Tomato_healthy": "TH",
}

# ====================== 加载模型 ======================
model = ModelC(num_classes=NUM_CLASSES)
state_dict = torch.load(WEIGHT_PATH, map_location=device)
model.load_state_dict(state_dict)
model.to(device)
model.eval()
print(f"Loaded weights: {WEIGHT_PATH}")

# ====================== 找到每个类别的一张样本 ======================
test_dir = os.path.join(DATA_ROOT, "test")
class_samples = {}
for cls_name in CLASS_NAMES:
    cls_path = os.path.join(test_dir, cls_name)
    imgs = [f for f in os.listdir(cls_path) if f.lower().endswith(('jpg', 'jpeg', 'png'))]
    if imgs:
        class_samples[cls_name] = os.path.join(cls_path, imgs[0])

# ====================== 预处理 ======================
val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

to_tensor = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
])

# ====================== Grad-CAM 多层热力图 ======================
target_layers_config = [
    (model.backbone.features[2], "After Conv 1"),
    (model.backbone.features[4], "After Conv 2"),
    (model.backbone.features[7], "After Conv 3"),
]

print(f"\nGenerating Grad-CAM heatmaps for {len(class_samples)} classes...")

for cls_idx, cls_name in enumerate(CLASS_NAMES):
    if cls_name not in class_samples:
        print(f"  Skipping {cls_name}: no test images")
        continue

    img_path = class_samples[cls_name]
    img_pil = Image.open(img_path).convert("RGB")
    input_tensor = val_transform(img_pil).unsqueeze(0).to(device)
    rgb_img = to_tensor(img_pil).permute(1, 2, 0).cpu().numpy()
    short = SHORT_NAMES.get(cls_name, cls_name)

    targets = [ClassifierOutputTarget(cls_idx)]

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(rgb_img)
    axes[0].set_title("Original Image", fontsize=12)
    axes[0].axis("off")

    for ax_idx, (target_layer, title) in enumerate(target_layers_config, start=1):
        cam = GradCAM(model=model, target_layers=[target_layer])
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
        grayscale_cam = grayscale_cam[0, :]
        cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
        axes[ax_idx].imshow(cam_image)
        axes[ax_idx].set_title(title, fontsize=12)
        axes[ax_idx].axis("off")

    fig.suptitle(f"{short} - {cls_name.replace('___', ' - ').replace('_', ' ')}", fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    save_path = os.path.join(OUTPUT_DIR, f"{short}_gradcam.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")

print(f"\nDone! All Grad-CAM heatmaps saved to {OUTPUT_DIR}/")
