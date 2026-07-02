import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns
import gc
import pandas as pd

# ====================== Matplotlib 中文 ======================
import matplotlib
matplotlib.use('Agg')
plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300

# ====================== 显存优化 ======================
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# ====================== 固定随机种子 ======================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
set_seed(42)

# ====================== 设备 ======================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# ====================== 全局配置 ======================
CONFIG = {
    "img_size": 300,
    "num_classes": 12,
    "batch_size": 8,
    "epochs": 20,
    "lr": 3e-4,
    "weight_decay": 1e-4,
    "num_workers": 4 if torch.cuda.is_available() else 0,
    "pin_memory": torch.cuda.is_available(),
    "data_root": r"/workspace/水果叶片",
    "save_path": r"./ablation_output",
    "backbones": ["Baseline", "Full_Model", "NoSE", "NoMixer"],
}
os.makedirs(CONFIG["save_path"], exist_ok=True)

# ====================== 数据集 ======================
class VegetableDataset(Dataset):
    def __init__(self, data_root, split="train", transform=None):
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
train_transform = transforms.Compose([
    transforms.Resize((320, 320)),
    transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), shear=10),
    transforms.RandomCrop(CONFIG["img_size"]),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
val_test_transform = transforms.Compose([
    transforms.Resize((CONFIG["img_size"], CONFIG["img_size"])),
    transforms.CenterCrop(CONFIG["img_size"]),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ====================== 数据加载（消融实验不使用样本不平衡策略）=====================
train_dataset = VegetableDataset(CONFIG["data_root"], "train", train_transform)
test_dataset = VegetableDataset(CONFIG["data_root"], "test", val_test_transform)
print(f"训练集数量: {len(train_dataset)}")
print(f"测试集数量: {len(test_dataset)}")
print(f"类别: {train_dataset.class_names}")

# 标准随机shuffle，无加权采样
dataloaders = {
    "train": DataLoader(train_dataset, CONFIG["batch_size"], shuffle=True,
                         num_workers=CONFIG["num_workers"], pin_memory=CONFIG["pin_memory"], drop_last=True),
    "test": DataLoader(test_dataset, CONFIG["batch_size"], shuffle=False,
                        num_workers=CONFIG["num_workers"], pin_memory=CONFIG["pin_memory"])
}

# ====================== 消融模型定义 ======================
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

class BaseEfficientNetModel(nn.Module):
    def __init__(self, num_classes=12):
        super().__init__()
        self.backbone = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.DEFAULT)
        self.in_features = self.backbone.classifier[1].in_features
    def forward(self, x):
        return torch.flatten(self.backbone.avgpool(self.backbone.features(x)), 1)

class EfficientNet_Baseline(BaseEfficientNetModel):
    """基线模型：原始 EfficientNet-B3"""
    def __init__(self, num_classes=12):
        super().__init__(num_classes)
        self.classifier = nn.Linear(self.in_features, num_classes)
    def forward(self, x):
        x = self.backbone.features(x)
        x = torch.flatten(self.backbone.avgpool(x), 1)
        return self.classifier(x)

class ModelC(BaseEfficientNetModel):
    """完整模型：SEBlock + MLP-Mixer"""
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

class ModelC_NoSE(BaseEfficientNetModel):
    """消融：去掉 SEBlock"""
    def __init__(self, num_classes=12):
        super().__init__(num_classes)
        self.proj = nn.Conv2d(self.in_features, 512, kernel_size=1)
        self.seq_len = 100
        self.mixer_layers = nn.ModuleList([
            MLPMixerLayer(dim=512, seq_len=self.seq_len) for _ in range(4)
        ])
        self.classifier = nn.Linear(512, num_classes)
    def forward(self, x):
        x = self.backbone.features(x)
        x = self.proj(x)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        for mixer in self.mixer_layers:
            x = mixer(x)
        x = x.mean(dim=1)
        return self.classifier(x)

class ModelC_NoMixer(BaseEfficientNetModel):
    """消融：去掉 MLP-Mixer，保留 SEBlock + GAP"""
    def __init__(self, num_classes=12):
        super().__init__(num_classes)
        self.se_block = SEBlock(self.in_features)
        self.classifier = nn.Linear(self.in_features, num_classes)
    def forward(self, x):
        x = self.backbone.features(x)
        x = self.se_block(x)
        x = torch.flatten(self.backbone.avgpool(x), 1)
        return self.classifier(x)

# ====================== 获取模型 ======================
def get_model(backbone_name, num_classes):
    if backbone_name == "Baseline":
        model = EfficientNet_Baseline(num_classes)
    elif backbone_name == "Full_Model":
        model = ModelC(num_classes)
    elif backbone_name == "NoSE":
        model = ModelC_NoSE(num_classes)
    elif backbone_name == "NoMixer":
        model = ModelC_NoMixer(num_classes)
    else:
        raise ValueError("仅支持 Baseline / Full_Model / NoSE / NoMixer")
    return model.to(device)

# ====================== 训练器（标准交叉熵，无加权）=====================
class ModelTrainer:
    def __init__(self, backbone, num_classes, config):
        self.backbone = backbone
        self.num_classes = num_classes
        self.config = config
        self.model = get_model(backbone, num_classes)
        self.save_dir = os.path.join(config["save_path"], backbone)
        os.makedirs(self.save_dir, exist_ok=True)
        self.criterion = nn.CrossEntropyLoss()  # 标准交叉熵，无加权
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', factor=0.5, patience=5)

    def train_one_epoch(self, loader):
        self.model.train()
        total_loss, total_correct = 0, 0
        all_preds, all_labels = [], []
        for imgs, labels in tqdm(loader, desc=f"Train {self.backbone}"):
            imgs, labels = imgs.to(device), labels.to(device)
            self.optimizer.zero_grad()
            outputs = self.model(imgs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
        acc = accuracy_score(all_labels, all_preds)
        return total_loss / len(loader), acc

    def evaluate(self, loader):
        self.model.eval()
        all_preds, all_labels = [], []
        total_loss = 0.0
        with torch.no_grad():
            for imgs, labels in loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = self.model(imgs)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        acc = accuracy_score(all_labels, all_preds)
        prec = precision_score(all_labels, all_preds, average="macro", zero_division=0)
        recall = recall_score(all_labels, all_preds, average="macro", zero_division=0)
        f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
        return total_loss / len(loader), acc, prec, recall, f1, all_labels, all_preds

    def run(self):
        train_loader = dataloaders["train"]
        test_loader = dataloaders["test"]
        best_test_acc = 0
        history = {"train_loss": [], "train_acc": []}
        last_lr = self.optimizer.param_groups[0]["lr"]
        for epoch in range(self.config["epochs"]):
            train_loss, train_acc = self.train_one_epoch(train_loader)
            test_loss, test_acc, test_prec, test_recall, test_f1, _, _ = self.evaluate(test_loader)
            self.scheduler.step(test_acc)
            current_lr = self.optimizer.param_groups[0]["lr"]
            lr_change = f" (↓{last_lr / current_lr:.1f}x)" if current_lr < last_lr else ""
            last_lr = current_lr
            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            print(f"[{self.backbone}] E{epoch+1:2d} | loss:{train_loss:.3f} | train_acc:{train_acc:.4f} | test_acc:{test_acc:.4f} | LR:{current_lr:.2e}{lr_change}")
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                torch.save(self.model.state_dict(), os.path.join(self.save_dir, "best.pth"))
        # 最终评估
        self.model.load_state_dict(torch.load(os.path.join(self.save_dir, "best.pth"), map_location=device))
        test_loss, test_acc, test_prec, test_recall, test_f1, y_true, y_pred = self.evaluate(test_loader)

        # 保存训练曲线
        plt.figure(figsize=(12,4))
        plt.subplot(121); plt.plot(history["train_loss"], label="训练损失"); plt.title(f"{self.backbone} 损失曲线")
        plt.subplot(122); plt.plot(history["train_acc"], label="训练准确率"); plt.title("准确率曲线")
        plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, "curve.png"), dpi=300)
        plt.close()

        return {
            "backbone": self.backbone,
            "acc": round(test_acc,4),
            "prec": round(test_prec,4),
            "recall": round(test_recall,4),
            "f1": round(test_f1,4)
        }

# ====================== 主运行 ======================
if __name__ == "__main__":
    results = []
    for backbone in CONFIG["backbones"]:
        print("\n" + "="*60)
        print(f"  训练模型: {backbone}")
        print("="*60)
        trainer = ModelTrainer(backbone, CONFIG["num_classes"], CONFIG)
        res = trainer.run()
        results.append(res)
        del trainer
        gc.collect()
        torch.cuda.empty_cache()

    df = pd.DataFrame(results)
    print("\n" + "="*70)
    print("                消融模型对比结果")
    print("="*70)
    print(df)
    df.to_csv("./ablation_model_comparison.csv", index=False, encoding="utf-8-sig")

    # 对比柱状图
    plt.figure(figsize=(16,10))
    metrics = ["acc", "prec", "recall", "f1"]
    for i, metric in enumerate(metrics):
        plt.subplot(2,2,i+1)
        sns.barplot(x="backbone", y=metric, data=df)
        plt.title(f"模型 {metric.upper()} 对比")
        plt.ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig("./ablation_model_comparison.png", dpi=300)
    plt.close()
    print("\n消融实验对比图已保存: ./ablation_model_comparison.png")
