import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image
from tqdm import tqdm


# 配置参数
class Config:
    # 数据参数
    train_dir = "../data/my_dataset/train"
    val_dir = "../data/my_dataset/val"
    num_classes = 7
    img_size = (512, 512)  # 统一输入尺寸

    # 训练参数
    batch_size = 32
    num_epochs = 20
    lr = 3e-4
    patience = 5

    # 蒸馏参数
    temp = 3.0  # 温度参数
    alpha = 0.7  # 软目标权重
    student_type = "resnet34"  # 可选resnet18/resnet34


# 数据集类（修复类别索引问题）
class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.class_to_idx = {}

        # 优先处理OK类别
        ok_dir = os.path.join(root_dir, "OK")
        if os.path.exists(ok_dir):
            self._add_class("OK", ok_dir)

        # 处理NG子类
        ng_dir = os.path.join(root_dir, "NG")
        if os.path.exists(ng_dir):
            for sub_dir in sorted(os.listdir(ng_dir)):  # sorted保证顺序一致
                self._add_class(sub_dir, os.path.join(ng_dir, sub_dir))

    def _add_class(self, class_name, dir_path):
        if class_name not in self.class_to_idx:
            self.class_to_idx[class_name] = len(self.class_to_idx)

        class_idx = self.class_to_idx[class_name]
        for img_name in os.listdir(dir_path):
            img_path = os.path.join(dir_path, img_name)
            if self._is_valid(img_path):
                self.image_paths.append(img_path)
                self.labels.append(class_idx)

    def _is_valid(self, img_path):
        try:
            Image.open(img_path).verify()
            return True
        except:
            print(f"Invalid image: {img_path}")
            return False

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]


# 数据增强
def get_transforms():
    return transforms.Compose([
        transforms.Resize(Config.img_size),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2, 0.2, 0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


# 模型构建
class ModelBuilder:
    @staticmethod
    def build_teacher():
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        model.fc = nn.Linear(model.fc.in_features, Config.num_classes)
        return model

    @staticmethod
    def build_student():
        if Config.student_type == "resnet34":
            model = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
        else:
            model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        model.fc = nn.Linear(model.fc.in_features, Config.num_classes)
        return model


# 自蒸馏训练器
class DistillationTrainer:
    def __init__(self, teacher, student, device):
        self.teacher = teacher.to(device).eval()
        self.student = student.to(device)
        self.device = device

        # 冻结教师模型
        for param in self.teacher.parameters():
            param.requires_grad = False

    def compute_loss(self, s_logits, t_logits, labels):
        # 软目标损失
        soft_loss = F.kl_div(
            F.log_softmax(s_logits / Config.temp, dim=1),
            F.softmax(t_logits / Config.temp, dim=1),
            reduction="batchmean"
        ) * (Config.temp ** 2)

        # 硬目标损失
        hard_loss = F.cross_entropy(s_logits, labels)

        return Config.alpha * soft_loss + (1 - Config.alpha) * hard_loss

    def train_epoch(self, loader, optimizer):
        self.student.train()
        total_loss = 0.0

        for images, labels in tqdm(loader, desc="Training"):
            images = images.to(self.device)
            labels = labels.to(self.device)

            # 教师预测
            with torch.no_grad():
                t_logits = self.teacher(images)

            # 学生预测
            s_logits = self.student(images)

            # 计算损失
            loss = self.compute_loss(s_logits, t_logits, labels)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images.size(0)

        return total_loss / len(loader.dataset)

    @torch.no_grad()
    def evaluate(self, loader):
        self.student.eval()
        total_loss = 0.0
        correct = 0

        for images, labels in tqdm(loader, desc="Evaluating"):
            images = images.to(self.device)
            labels = labels.to(self.device)

            t_logits = self.teacher(images)
            s_logits = self.student(images)

            loss = self.compute_loss(s_logits, t_logits, labels)
            total_loss += loss.item() * images.size(0)

            _, preds = torch.max(s_logits, 1)
            correct += (preds == labels).sum().item()

        avg_loss = total_loss / len(loader.dataset)
        accuracy = 100. * correct / len(loader.dataset)
        return avg_loss, accuracy


# 主流程
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 数据加载
    transform = get_transforms()
    train_set = CustomDataset(Config.train_dir, transform)
    val_set = CustomDataset(Config.val_dir, transform)

    train_loader = DataLoader(train_set, batch_size=Config.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=Config.batch_size, shuffle=False, num_workers=4)

    # 第一阶段：训练教师模型
    print("\n=== Training Teacher Model ===")
    teacher = ModelBuilder.build_teacher().to(device)
    optimizer = optim.AdamW(teacher.parameters(), lr=Config.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "max", patience=2)

    best_acc = 0.0
    patience_counter = 0

    for epoch in range(Config.num_epochs):
        # 训练
        teacher.train()
        total_loss = 0.0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}"):
            images, labels = images.to(device), labels.to(device)

            outputs = teacher(images)
            loss = F.cross_entropy(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images.size(0)

        # 验证
        teacher.eval()
        correct = 0
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = teacher(images)
            correct += (outputs.argmax(1) == labels).sum().item()

        val_acc = 100 * correct / len(val_set)
        print(f"Epoch {epoch + 1}: Loss={total_loss / len(train_set):.4f}, Val Acc={val_acc:.2f}%")

        # 早停机制
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(teacher.state_dict(), "best_teacher.pth")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= Config.patience:
                print("Early stopping triggered!")
                break

        scheduler.step(val_acc)

    # 第二阶段：自蒸馏训练
    print("\n=== Distilling Student Model ===")
    teacher.load_state_dict(torch.load("best_teacher.pth"))
    student = ModelBuilder.build_student().to(device)

    distiller = DistillationTrainer(teacher, student, device)
    optimizer = optim.AdamW(student.parameters(), lr=Config.lr * 0.1)

    best_student_acc = 0.0
    patience_counter = 0

    for epoch in range(Config.num_epochs):
        # 训练
        train_loss = distiller.train_epoch(train_loader, optimizer)

        # 验证
        val_loss, val_acc = distiller.evaluate(val_loader)
        print(f"Epoch {epoch + 1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, Val Acc={val_acc:.2f}%")

        # 保存最佳学生模型
        if val_acc > best_student_acc:
            best_student_acc = val_acc
            torch.save(student.state_dict(), "best_student.pth")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= Config.patience:
                print("Early stopping distillation!")
                break


if __name__ == "__main__":
    main()