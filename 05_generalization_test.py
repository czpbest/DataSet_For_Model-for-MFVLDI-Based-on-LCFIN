import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ====================== 设备 ======================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# ====================== 配置 ======================
DATA_ROOT = r"/workspace/水果叶片"
MODEL_PATH = r"./train_output/best_model.pth"

CONFIG = {
    "img_size": 300,
    "batch_size": 8,
    "lr": 1e-3,
    "epochs": 5,
    "num_workers": 4 if torch.cuda.is_available() else 0,
}

# ====================== 万能数据集 ======================
class UniversalDataset(Dataset):
    def __init__(self, data_root, split="train", transform=None):
        self.data_root = data_root
        self.split = split
        self.transform = transform
        self.split_dir = os.path.join(data_root, split)
        self.class_names = sorted([d for d in os.listdir(self.split_dir) if os.path.isdir(os.path.join(self.split_dir, d))])
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.class_names)}
        self.num_classes = len(self.class_names)
        self.image_paths = []
        self.labels = []
        for cls in self.class_names:
            cls_dir = os.path.join(self.split_dir, cls)
            for img in os.listdir(cls_dir):
                if img.lower().endswith(('jpg', 'jpeg', 'png')):
                    self.image_paths.append(os.path.join(cls_dir, img))
                    self.labels.append(self.class_to_idx[cls])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path, label = self.image_paths[idx], self.labels[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label

# ====================== 预处理 ======================
train_transform = transforms.Compose([
    transforms.Resize((320, 320)),
    transforms.RandomCrop(300),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.CenterCrop(300),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ====================== 模型定义（和训练时完全一致）=====================
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

class ModelC(nn.Module):
    def __init__(self, num_classes=12):
        super().__init__()
        self.backbone = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.DEFAULT)
        self.out_channels = self.backbone.classifier[1].in_features
        self.se_block = SEBlock(self.out_channels)
        self.proj = nn.Conv2d(self.out_channels, 512, kernel_size=1)
        self.mixer_layers = nn.ModuleList([MLPMixerLayer(512, 100) for _ in range(4)])
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.backbone.features(x)
        x = self.se_block(x)
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        for layer in self.mixer_layers:
            x = layer(x)
        x = x.mean(1)
        return self.classifier(x)

# ====================== 加载数据 ======================
train_dataset = UniversalDataset(DATA_ROOT, "train", train_transform)
test_dataset = UniversalDataset(DATA_ROOT, "test", test_transform)
NUM_CLASSES = train_dataset.num_classes

train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True, num_workers=CONFIG["num_workers"])
test_loader = DataLoader(test_dataset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=CONFIG["num_workers"])

print(f"训练集: {len(train_dataset)}")
print(f"测试集: {len(test_dataset)}")
print(f"类别数: {NUM_CLASSES} | {train_dataset.class_names}")

# ====================== 加载模型 ======================
model = ModelC(num_classes=12).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))

# ====================== 冻结全部，只微调分类头 ======================
for param in model.parameters():
    param.requires_grad = False

model.classifier = nn.Linear(512, NUM_CLASSES).to(device)
for param in model.classifier.parameters():
    param.requires_grad = True

# ====================== 训练（只微调分类头）=====================
optimizer = torch.optim.Adam(model.classifier.parameters(), lr=CONFIG["lr"])
criterion = nn.CrossEntropyLoss()
best_acc = 0

for epoch in range(CONFIG["epochs"]):
    model.train()
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']}")

    for imgs, labels in pbar:
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        pbar.set_postfix(loss=f"{loss.item():.3f}")

    # 测试
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(device)
            preds = torch.argmax(model(imgs), dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    acc = accuracy_score(all_labels, all_preds)
    print(f"Epoch {epoch+1} 准确率: {acc:.4f}")

    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), "./best_finetune.pth")

# ====================== 最终结果 ======================
model.load_state_dict(torch.load("./best_finetune.pth", map_location=device))
model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for imgs, labels in test_loader:
        imgs = imgs.to(device)
        preds = torch.argmax(model(imgs), dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

acc = accuracy_score(all_labels, all_preds)
prec = precision_score(all_labels, all_preds, average="macro", zero_division=0)
recall = recall_score(all_labels, all_preds, average="macro", zero_division=0)
f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

print("\n" + "="*60)
print("           泛化实验（只微调分类头）")
print("="*60)
print(f"准确率 Accuracy:  {acc:.4f}")
print(f"精确率 Precision: {prec:.4f}")
print(f"召回率 Recall:    {recall:.4f}")
print(f"F1分数 F1-Score:  {f1:.4f}")
print("="*60)
