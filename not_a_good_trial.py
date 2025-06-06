import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
from PIL import Image
import torch.nn.functional as F
import numpy as np
import timm

# Mixup 数据增强函数
def mixup_data(x, y, indices, alpha=0.2):
    """
    返回混合样本和对应标签/索引：
    mixed_x, y_a, y_b, idx_a, idx_b, lam
    """
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    batch_size = x.size(0)
    perm = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[perm]
    y_a, y_b = y, y[perm]
    idx_a, idx_b = indices, indices[perm]
    return mixed_x, y_a, y_b, idx_a, idx_b, lam

# TinyViT-5M 学生模型定义
class TinyViT5M(nn.Module):
    def __init__(self, num_classes=7, drop_rate=0.1):
        super(TinyViT5M, self).__init__()
        self.backbone = timm.create_model(
            'tiny_vit_5m_224', pretrained=False,
            num_classes=num_classes,
            drop_rate=drop_rate,
            drop_path_rate=0.0
        )
        self.dropout = nn.Dropout(drop_rate)

    def forward(self, x, return_features=False):
        features = self.backbone.forward_features(x)
        if return_features:
            return features
        x = self.dropout(features)
        return self.backbone.head(x)

# 自定义数据集类
class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        super(CustomDataset, self).__init__()
        self.transform = transform
        self.image_paths, self.labels = [], []
        self.class_to_idx = {}
        for category in os.listdir(root_dir):
            path_cat = os.path.join(root_dir, category)
            if not os.path.isdir(path_cat):
                continue
            if category == 'NG':
                for idx, sub in enumerate(os.listdir(path_cat)):
                    sub_dir = os.path.join(path_cat, sub)
                    if not os.path.isdir(sub_dir):
                        continue
                    self.class_to_idx[sub] = idx
                    for img in os.listdir(sub_dir):
                        self.image_paths.append(os.path.join(sub_dir, img))
                        self.labels.append(idx)
            elif category == 'OK':
                ok_idx = len(self.class_to_idx)
                self.class_to_idx['OK'] = ok_idx
                for img in os.listdir(path_cat):
                    self.image_paths.append(os.path.join(path_cat, img))
                    self.labels.append(ok_idx)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx], idx

# 蒸馏器定义
class Distiller(nn.Module):
    def __init__(self, student, teacher_logits, index_mapping,
                 alpha=0.7, temperature=4.0, label_smoothing=0.1):
        super(Distiller, self).__init__()
        self.student = student
        self.register_buffer('teacher_logits', teacher_logits)
        self.register_buffer('index_mapping', index_mapping)
        self.alpha = alpha
        self.temperature = temperature
        self.ce_loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.kd_loss = nn.KLDivLoss(reduction='batchmean')

    def forward(self, images, labels, indices):
        device = self.teacher_logits.device
        images = images.to(device)
        labels = labels.to(device)
        indices = indices.to(device)

        outputs = self.student(images)
        loss_ce = self.ce_loss(outputs, labels)

        mapped_idx = self.index_mapping[indices]
        if mapped_idx.max() >= self.teacher_logits.size(0) or mapped_idx.min() < 0:
            raise IndexError(f"Mapped idx out of bounds: {mapped_idx.min()}..{mapped_idx.max()}")
        t_logits = self.teacher_logits[mapped_idx]
        if t_logits.size(1) != outputs.size(1):
            raise ValueError(f"Teacher logits dim {t_logits.size(1)} != student dim {outputs.size(1)}")

        T = self.temperature
        teacher_prob = F.softmax(t_logits / T, dim=1)
        student_log_prob = F.log_softmax(outputs / T, dim=1)
        loss_kd = self.kd_loss(student_log_prob, teacher_prob) * (T * T)

        loss = self.alpha * loss_kd + (1 - self.alpha) * loss_ce
        return outputs, loss

    def fit(self, train_loader, val_loader, num_epochs=20,
            lr=5e-5, weight_decay=1e-3, mixup_alpha=0.2, max_grad_norm=1.0, device=None):
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(device)

        optimizer = optim.AdamW(self.student.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

        for epoch in range(num_epochs):
            self.student.train()
            total_loss = 0.0
            for images, labels, indices in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
                images = images.to(device)
                labels = labels.to(device)
                indices = indices.to(device)

                # Mixup 增强
                mixed_x, y_a, y_b, idx_a, idx_b, lam = mixup_data(images, labels, indices, alpha=mixup_alpha)

                optimizer.zero_grad()
                outputs = self.student(mixed_x)

                # 混合交叉熵损失
                loss_ce = lam * self.ce_loss(outputs, y_a) + (1 - lam) * self.ce_loss(outputs, y_b)

                # 混合教师 logits
                t_logits_a = self.teacher_logits[self.index_mapping[idx_a]]
                t_logits_b = self.teacher_logits[self.index_mapping[idx_b]]
                t_logits = lam * t_logits_a + (1 - lam) * t_logits_b

                # 蒸馏损失
                T = self.temperature
                teacher_prob = F.softmax(t_logits / T, dim=1)
                student_log_prob = F.log_softmax(outputs / T, dim=1)
                loss_kd = self.kd_loss(student_log_prob, teacher_prob) * (T * T)

                loss = self.alpha * loss_kd + (1 - self.alpha) * loss_ce
                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.student.parameters(), max_grad_norm)
                optimizer.step()
                total_loss += loss.item()

            avg_train_loss = total_loss / len(train_loader)
            print(f'Epoch {epoch+1} Train Loss: {avg_train_loss:.4f}')
            scheduler.step()

            # 验证阶段
            self.student.eval()
            val_loss = 0.0
            correct = total = 0
            with torch.no_grad():
                for images, labels, indices in val_loader:
                    images = images.to(device)
                    labels = labels.to(device)
                    outputs = self.student(images)
                    val_loss += self.ce_loss(outputs, labels).item()
                    preds = outputs.argmax(dim=1)
                    correct += (preds == labels).sum().item()
                    total += labels.size(0)
            avg_val_loss = val_loss / len(val_loader)
            val_acc = 100 * correct / total
            print(f'Epoch {epoch+1} Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%')

        print('Training Complete.')


def main():
    data = torch.load('teacher_logits.pt')
    teacher_logits = data['logits']
    index_mapping = data['index_mapping']

    transform = transforms.Compose([
        transforms.Resize((1224, 1024)),
        transforms.RandomResizedCrop(256, scale=(0.8,1.0), ratio=(0.9,1.1)),
        transforms.GaussianBlur(3, sigma=(0.1,0.5)),
        transforms.RandomAdjustSharpness(0.5, p=0.5),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    train_ds = CustomDataset(os.path.join('data','my_dataset','train'), transform)
    val_ds = CustomDataset(os.path.join('data','my_dataset','val'), transform)

    train_loader = DataLoader(
        train_ds, batch_size=16, shuffle=True, num_workers=0,
        collate_fn=lambda batch: (
            torch.stack([item[0] for item in batch]),
            torch.tensor([item[1] for item in batch]),
            torch.tensor([item[2] for item in batch])
        )
    )
    val_loader = DataLoader(
        val_ds, batch_size=16, shuffle=False, num_workers=0,
        collate_fn=lambda batch: (
            torch.stack([item[0] for item in batch]),
            torch.tensor([item[1] for item in batch]),
            torch.tensor([item[2] for item in batch])
        )
    )

    distiller = Distiller(
        student=TinyViT5M(num_classes=7),
        teacher_logits=teacher_logits,
        index_mapping=index_mapping
    )
    distiller.fit(train_loader, val_loader, num_epochs=20)

if __name__ == '__main__':
    main()