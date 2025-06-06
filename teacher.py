import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from PIL import Image
from torchvision import models, transforms
from torchvision.models import ResNet50_Weights
from tqdm import tqdm
from torch.utils.data import SequentialSampler  # 新增导入

def save_teacher_logits(teacher_model):

    # 检查数据集有效性
    if len(train_dataset) == 0:
        raise ValueError("训练数据集为空，请检查数据路径和过滤逻辑")

    # 加载模型并移动到设备（修复重复加载问题）
    teacher_model.load_state_dict(torch.load('88best_teacher_model.pth'))  # 只保留一次加载
    teacher_model.to(device)
    teacher_model.eval()

    # 冻结所有参数
    for param in teacher_model.parameters():
        param.requires_grad = False

    # 创建带索引的数据加载器（关键修改）
    index_loader = data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        sampler=SequentialSampler(train_dataset)  # 显式顺序采样
    )

    # 保存 logits、标签和索引
    all_logits = []
    all_labels = []
    all_indices = []  # 新增索引收集

    with torch.no_grad():
        for batch in tqdm(index_loader, desc="生成Logits"):
            images, labels, indices = batch  # 获取索引（需修改数据集类）
            images = images.to(device)
            logits = teacher_model(images)
            all_logits.append(logits.cpu())
            all_labels.append(labels)
            all_indices.append(indices)  # 保存原始索引

    # 合并数据并保存
    logits_tensor = torch.cat(all_logits)
    labels_tensor = torch.cat(all_labels)
    indices_tensor = torch.cat(all_indices)  # 合并索引

    # 创建索引映射表（关键修改）
    index_mapping = torch.zeros(len(train_dataset), dtype=torch.long)
    global_idx = 0
    for batch_indices in all_indices:
        for relative_pos, original_idx in enumerate(batch_indices.tolist()):
            index_mapping[original_idx] = global_idx + relative_pos
        global_idx += len(batch_indices)

    # 保存数据
    torch.save({
        'logits': logits_tensor,
        'labels': labels_tensor,
        'index_mapping': index_mapping,
        'class_to_idx': train_dataset.class_to_idx
    }, 'teacher_logits.pt')

    print("Logits 和标签已保存至 teacher_logits.pt")

# 定义数据集类
class CustomDataset(data.Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.class_to_idx = {}

        # 收集所有图像路径和标签
        for category in os.listdir(root_dir):
            category_dir = os.path.join(root_dir, category)
            if os.path.isdir(category_dir):
                if category == 'NG':
                    for class_idx, ng_subclass in enumerate(os.listdir(category_dir)):
                        ng_subclass_dir = os.path.join(category_dir, ng_subclass)
                        if os.path.isdir(ng_subclass_dir):
                            self.class_to_idx[ng_subclass] = class_idx
                            for image_name in os.listdir(ng_subclass_dir):
                                image_path = os.path.join(ng_subclass_dir, image_name)
                                if self._is_valid_image(image_path):
                                    self.image_paths.append(image_path)
                                    self.labels.append(class_idx)
                elif category == 'OK':
                    ok_class_idx = len(self.class_to_idx)
                    self.class_to_idx['OK'] = ok_class_idx
                    for image_name in os.listdir(category_dir):
                        image_path = os.path.join(category_dir, image_name)
                        if self._is_valid_image(image_path):
                            self.image_paths.append(image_path)
                            self.labels.append(ok_class_idx)

    def _is_valid_image(self, image_path):
        try:
            with Image.open(image_path) as img:
                img.verify()
            return True
        except:
            print(f"Invalid image file: {image_path}")
            return False

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        try:
            image = Image.open(image_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, label, idx  # 新增返回索引
        except Exception as e:
            dummy_image = torch.randn(3, 224, 224)
            return dummy_image, label, idx  # 保持索引返回

# 超参数和路径
train_dataset_dir = os.path.join('data', 'my_dataset', 'train')
val_dataset_dir = os.path.join('data', 'my_dataset', 'val')
num_classes = 7
batch_size = 32
num_epochs = 20
learning_rate = 0.0005
num_workers = 4

# 数据预处理和加载
data_transform = transforms.Compose([
    transforms.Resize((1224, 1024)),
    transforms.RandomResizedCrop(256, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5)),
    transforms.RandomAdjustSharpness(sharpness_factor=0.5, p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 创建数据集实例
train_dataset = CustomDataset(train_dataset_dir, transform=data_transform)
val_dataset = CustomDataset(val_dataset_dir, transform=data_transform)

# 如果数据集为空，可能路径有误
if len(train_dataset) == 0 or len(val_dataset) == 0:
    raise FileNotFoundError(f"No valid data found in {train_dataset_dir} or {val_dataset_dir}. Please check the path.")

def collate_fn(batch):
    images = torch.stack([item[0] for item in batch])
    labels = torch.tensor([item[1] for item in batch])
    indices = torch.tensor([item[2] for item in batch])
    return images, labels, indices

# 创建数据加载器
train_loader = data.DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    collate_fn=collate_fn  # 添加自定义打包函数
)

val_loader = data.DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    collate_fn=collate_fn  # 添加自定义打包函数
)
# 定义教师模型
weights = ResNet50_Weights.DEFAULT
teacher_model = models.resnet50(weights=weights)
teacher_model.fc = nn.Linear(teacher_model.fc.in_features, num_classes)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(teacher_model.parameters(), lr=learning_rate)

# 学习率调度器
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

# 早停机制
best_val_acc = 0.0
patience = 3
patience_counter = 0

# 训练教师模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
teacher_model.to(device)

if __name__ == '__main__':
    for epoch in range(num_epochs):
        # 训练阶段
        teacher_model.train()
        running_loss = 0.0
        for images, labels, _ in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = teacher_model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
        epoch_loss = running_loss / len(train_dataset)
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {epoch_loss:.4f}')

        # 验证阶段
        teacher_model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels, _ in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = teacher_model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_loss = val_loss / len(val_dataset)
        val_acc = 100 * correct / total
        print(f'Epoch [{epoch+1}/{num_epochs}], Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')

        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(teacher_model.state_dict(), '88best_teacher_model.pth')
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print('Early stopping triggered.')
                break

        # 调整学习率
        scheduler.step()

    print('Training completed.')
    # 训练结束后，保存logits
    teacher_model_for_inference = models.resnet50(weights=weights)
    teacher_model_for_inference.fc = nn.Linear(teacher_model_for_inference.fc.in_features, num_classes)
    teacher_model_for_inference.load_state_dict(torch.load('88best_teacher_model.pth'))
    teacher_model_for_inference.to(device)
    save_teacher_logits(teacher_model_for_inference)
