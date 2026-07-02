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
from collections import defaultdict

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

# ====================== 全局配置 ======================
CONFIG = {
    "img_size": 300,
    "num_classes": 12,
    "batch_size": 8,
    "num_workers": 4 if torch.cuda.is_available() else 0,
    "pin_memory": torch.cuda.is_available(),
    "data_root": r"/workspace/水果叶片",
    "save_path": r"./violin_output",
    "best_model_path": r"./train_output/best_model.pth",
    "num_runs": 5,
    "num_sub_evals": 10,
}
os.makedirs(CONFIG["save_path"], exist_ok=True)

# ====================== 类别简称映射 ======================
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

def get_short_names(class_names):
    return [SHORT_NAMES.get(n, n) for n in class_names]

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
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
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

# ====================== 加载数据 ======================
train_dataset = VegetableDataset(CONFIG["data_root"], "train", train_transform)
test_dataset = VegetableDataset(CONFIG["data_root"], "test", val_test_transform)
print(f"训练集数量: {len(train_dataset)}")
print(f"测试集数量: {len(test_dataset)}")
print(f"类别: {train_dataset.class_names}")

dataloaders = {
    "train": DataLoader(train_dataset, CONFIG["batch_size"], shuffle=True,
                        num_workers=CONFIG["num_workers"], pin_memory=CONFIG["pin_memory"], drop_last=True),
    "test": DataLoader(test_dataset, CONFIG["batch_size"], shuffle=False,
                       num_workers=CONFIG["num_workers"], pin_memory=CONFIG["pin_memory"])
}

# ====================== 模型获取 ======================
def get_model(num_classes):
    model = ModelC(num_classes=num_classes)
    state_dict = torch.load(CONFIG["best_model_path"], map_location=device)
    model.load_state_dict(state_dict)
    print(f"已加载最佳模型权重: {CONFIG['best_model_path']}")
    return model.to(device)

# ====================== 带轻度增强的测试transform ======================
def get_perturbed_transform(img_size):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomAffine(degrees=5, translate=(0.02, 0.02)),
        transforms.CenterCrop(img_size),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

# ====================== 评估函数 ======================
def evaluate_model(model, loader, criterion):
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0.0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, average="macro", zero_division=0)
    recall = recall_score(all_labels, all_preds, average="macro", zero_division=0)
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    cm = confusion_matrix(all_labels, all_preds)
    per_class_acc = cm.diagonal() / cm.sum(axis=1)

    return {
        "loss": total_loss / len(loader),
        "acc": acc,
        "prec": prec,
        "recall": recall,
        "f1": f1,
        "per_class_acc": per_class_acc,
        "y_true": all_labels,
        "y_pred": all_preds,
    }

# ====================== 小提琴图绘制 ======================
def plot_violin(results, class_names, save_path="./violin_all.png"):
    short_names = get_short_names(class_names)

    all_class_accs = []
    labels = []
    for res in results:
        for cls_idx, cls_acc in enumerate(res["per_class_acc"]):
            all_class_accs.append(cls_acc)
            labels.append(short_names[cls_idx])
    df_cls = pd.DataFrame({"acc": all_class_accs, "class": labels})

    class_palette = {
        "AAS": "#e74c3c", "ABR": "#3498db", "ACAR": "#2ecc71", "AH": "#f39c12",
        "GBR": "#9b59b6", "GEBM": "#1abc9c", "GLBILS": "#e67e22", "GH": "#34495e",
        "TBS": "#e91e63", "TEB": "#00bcd4", "TLM": "#8bc34a", "TH": "#ff9800",
    }

    all_cls = ["AAS", "ABR", "ACAR", "AH", "GBR", "GEBM", "GLBILS", "GH", "TBS", "TEB", "TLM", "TH"]
    fig, ax = plt.subplots(figsize=(28, 6))
    sns.violinplot(x="class", y="acc", hue="class", data=df_cls,
                   order=all_cls, palette=class_palette, legend=False, ax=ax)
    ax.set_xlabel("")
    ax.set_ylabel("Accuracy", fontsize=14)
    ax.set_title("Per-Class Accuracy Distribution (Violin Plot)", fontsize=16)
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"小提琴图已保存: {save_path}")

# ====================== 主运行 ======================
if __name__ == "__main__":
    model = get_model(CONFIG["num_classes"])
    criterion = nn.CrossEntropyLoss()

    # ---- 先用标准测试集做一次完整评估 ----
    test_loader = dataloaders["test"]
    base_result = evaluate_model(model, test_loader, criterion)
    print(f"\n标准测试集结果: acc={base_result['acc']:.4f} | f1={base_result['f1']:.4f} | prec={base_result['prec']:.4f} | recall={base_result['recall']:.4f}")

    # ---- 小提琴实验：多次扰动评估 ----
    results = []
    num_runs = CONFIG["num_runs"]
    num_sub_evals = CONFIG["num_sub_evals"]

    for run_id in range(1, num_runs + 1):
        print(f"\n小提琴实验 - 第 {run_id}/{num_runs} 次运行")
        set_seed(42 + run_id)
        perturbed_transform = get_perturbed_transform(CONFIG["img_size"])
        perturbed_dataset = VegetableDataset(CONFIG["data_root"], "test", perturbed_transform)
        perturbed_loader = DataLoader(perturbed_dataset, CONFIG["batch_size"], shuffle=True,
                                       num_workers=CONFIG["num_workers"], pin_memory=CONFIG["pin_memory"])
        for sub_id in range(num_sub_evals):
            set_seed(42 + run_id * 100 + sub_id)
            res = evaluate_model(model, perturbed_loader, criterion)
            results.append(res)
            print(f"  sub_{sub_id+1}: acc={res['acc']:.4f} | f1={res['f1']:.4f}")

    all_accs = [r["acc"] for r in results]
    all_f1s = [r["f1"] for r in results]
    print(f"\n{'='*70}")
    print(f"  小提琴实验汇总 ({len(results)} 次评估)")
    print(f"  ACC:  mean={np.mean(all_accs):.4f}  std={np.std(all_accs):.4f}")
    print(f"  F1:   mean={np.mean(all_f1s):.4f}  std={np.std(all_f1s):.4f}")
    print(f"{'='*70}")

    # 保存CSV
    short_names = get_short_names(train_dataset.class_names)
    rows = []
    for i, r in enumerate(results):
        row = {
            "run": i+1,
            "acc": round(r["acc"],4),
            "prec": round(r["prec"],4),
            "recall": round(r["recall"],4),
            "f1": round(r["f1"],4),
            "loss": round(r["loss"],4),
        }
        for cls_idx, cls_acc in enumerate(r["per_class_acc"]):
            row[short_names[cls_idx]] = round(cls_acc, 4)
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv("./violin_results.csv", index=False, encoding="utf-8-sig")
    print(f"结果CSV已保存: ./violin_results.csv")

    # 绘制小提琴图
    plot_violin(results, train_dataset.class_names)
