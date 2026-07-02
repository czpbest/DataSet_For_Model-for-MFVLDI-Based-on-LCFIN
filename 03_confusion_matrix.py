import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300

from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns

# ====================== 设备 ======================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# ====================== 配置 ======================
WEIGHT_PATH = r"./train_output/best_model.pth"
DATA_ROOT = r"/workspace/水果叶片"
IMG_SIZE = 300
NUM_CLASSES = 12
OUTPUT_DIR = r"./confusion_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ====================== 网络模块 ======================
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

# ====================== 数据集 ======================
class VegetableDataset(Dataset):
    def __init__(self, data_root, split="test", transform=None):
        self.data_root = data_root
        self.split = split
        self.transform = transform
        self.split_dir = os.path.join(data_root, split)
        self.class_names = sorted([d for d in os.listdir(self.split_dir) if os.path.isdir(os.path.join(self.split_dir, d))])
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.class_names)}
        self.image_paths = []
        self.labels = []
        for cls_name in self.class_names:
            cls_path = os.path.join(self.split_dir, cls_name)
            for img_name in os.listdir(cls_path):
                if img_name.lower().endswith(('jpg', 'jpeg', 'png')):
                    self.image_paths.append(os.path.join(cls_path, img_name))
                    self.labels.append(self.class_to_idx[cls_name])
    def __len__(self):
        return len(self.image_paths)
    def __getitem__(self, idx):
        img_path, label = self.image_paths[idx], self.labels[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label

# ====================== 预处理 ======================
test_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ====================== 加载数据 ======================
test_dataset = VegetableDataset(DATA_ROOT, "test", test_transform)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=4 if torch.cuda.is_available() else 0)
print(f"测试集数量: {len(test_dataset)}")
print(f"类别: {test_dataset.class_names}")

# ====================== 加载模型 ======================
model = ModelC(NUM_CLASSES).to(device)
model.load_state_dict(torch.load(WEIGHT_PATH, map_location=device))
model.eval()
print(f"已加载模型: {WEIGHT_PATH}")

# ====================== 测试评估 ======================
all_preds, all_labels = [], []
with torch.no_grad():
    for imgs, labels in test_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

acc = accuracy_score(all_labels, all_preds)
prec = precision_score(all_labels, all_preds, average="macro", zero_division=0)
recall = recall_score(all_labels, all_preds, average="macro", zero_division=0)
f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

print("\n" + "="*60)
print("测试评估结果")
print("="*60)
print(f"Accuracy:  {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-Score:  {f1:.4f}")
print("="*60)

# ====================== 混淆矩阵 ======================
cm = confusion_matrix(all_labels, all_preds)

# 数值版
plt.figure(figsize=(14, 12))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=test_dataset.class_names, yticklabels=test_dataset.class_names,
            square=True, linewidths=0.5, cbar_kws={"shrink": 0.8, "label": "样本数量"})
plt.title('混淆矩阵 (Confusion Matrix)', fontsize=16, pad=20)
plt.xlabel('预测标签 (Predicted)', fontsize=12)
plt.ylabel('真实标签 (True)', fontsize=12)
plt.xticks(rotation=45, ha="right", fontsize=9)
plt.yticks(rotation=0, fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"), dpi=300, bbox_inches='tight')
plt.close()
print(f"\n混淆矩阵已保存: {OUTPUT_DIR}/confusion_matrix.png")

# 归一化版
cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
plt.figure(figsize=(14, 12))
sns.heatmap(cm_norm, annot=True, fmt=".2%", cmap="Blues",
            xticklabels=test_dataset.class_names, yticklabels=test_dataset.class_names,
            square=True, linewidths=0.5, vmin=0, vmax=1,
            cbar_kws={"shrink": 0.8, "label": "准确率"})
plt.title('归一化混淆矩阵 (Normalized Confusion Matrix)', fontsize=16, pad=20)
plt.xlabel('预测标签 (Predicted)', fontsize=12)
plt.ylabel('真实标签 (True)', fontsize=12)
plt.xticks(rotation=45, ha="right", fontsize=9)
plt.yticks(rotation=0, fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix_normalized.png"), dpi=300, bbox_inches='tight')
plt.close()
print(f"归一化混淆矩阵已保存: {OUTPUT_DIR}/confusion_matrix_normalized.png")
