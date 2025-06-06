import os
import shutil
import numpy as np
from PIL import Image
import imagehash
import cv2
from tqdm import tqdm

# ================== 配置参数 ==================
SOURCE_BASE_DIR = r"U:\\BaiduNetdiskDownload\\Johnson-Cognex-data"
SOURCE_NG_FOLDER_NAME = "NG Images"
SOURCE_OK_FOLDER_NAME = "OK Images"

DESTINATION_BASE_DIR = r".\\Johnson-Cognex-data"
DEST_NG_FOLDER_NAME = "NG"
DEST_OK_FOLDER_NAME = "OK"
DEST_CORRUPT_NAME = "Corrupt"
DEST_DUPLICATE_NAME = "Duplicate"
DEST_NO_NEEDLE_NAME = "No_Needle"

# 感知哈希去重阈值
HASH_THRESHOLD = 5

# 针线检测HSV范围 (青绿色针线)
NEEDLE_HSV_LOW = (100, 50, 50)
NEEDLE_HSV_HIGH = (140, 255, 255)

# 纯黑图片阈值
BLACK_THRESHOLD_MEAN = 30
BLACK_THRESHOLD_VAR = 10


# ================== 针线增强模块 ==================
def enhance_needle_details(img):
    """三阶段针线细节增强"""
    # 阶段1：非锐化掩蔽增强边缘
    blurred = cv2.GaussianBlur(img, (0, 0), 2.0)
    sharpened = cv2.addWeighted(img, 1.5, blurred, -0.5, 0)

    # 阶段2：自适应对比度增强
    lab = cv2.cvtColor(sharpened, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # 基于亮度分布自动调整CLAHE参数
    l_mean = np.mean(l)
    clip_limit = 2.0 if l_mean > 100 else 4.0
    grid_size = 8 if l_mean > 100 else 4

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(grid_size, grid_size))
    l_enhanced = clahe.apply(l)

    # 阶段3：保留边缘的降噪
    lab_enhanced = cv2.merge((l_enhanced, a, b))
    enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

    # 导向滤波优化
    return cv2.ximgproc.guidedFilter(enhanced, enhanced, radius=5, eps=0.01)


# ================== 核心功能模块 ==================
def is_corrupt_image(img_path):
    """检测损坏图片"""
    try:
        img = Image.open(img_path)
        img.verify()
        img = Image.open(img_path)
        img.load()
        return False
    except:
        return True


def compute_image_hash(img_path):
    """计算感知哈希值"""
    try:
        with Image.open(img_path) as img:
            return imagehash.phash(img)
    except:
        return None


def detect_needle_presence(img_path):
    """检测针线是否存在"""
    try:
        img = cv2.imread(img_path)
        if img is None:
            return False

        # 检查纯黑图片
        if np.mean(img) < BLACK_THRESHOLD_MEAN and np.var(img) < BLACK_THRESHOLD_VAR:
            return False

        # HSV颜色空间检测
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, NEEDLE_HSV_LOW, NEEDLE_HSV_HIGH)

        # 计算针线区域占比
        needle_ratio = cv2.countNonZero(mask) / (img.shape[0] * img.shape[1])
        return needle_ratio > 0.05
    except:
        return False


# ================== 主处理流程 ==================
def process_images(source_base, source_ng_name, source_ok_name, dest_base,
                   dest_ng_name, dest_ok_name, enhance_details=False):
    # 创建所有目标目录
    os.makedirs(os.path.join(dest_base, dest_ng_name), exist_ok=True)
    os.makedirs(os.path.join(dest_base, dest_ok_name), exist_ok=True)
    os.makedirs(os.path.join(dest_base, DEST_CORRUPT_NAME), exist_ok=True)
    os.makedirs(os.path.join(dest_base, DEST_DUPLICATE_NAME), exist_ok=True)
    os.makedirs(os.path.join(dest_base, DEST_NO_NEEDLE_NAME), exist_ok=True)

    # 初始化计数器
    counters = {
        'ng': 0, 'ok': 0, 'corrupt': 0,
        'duplicate': 0, 'no_needle': 0, 'other': 0,
        'enhanced': 0
    }

    # 哈希值记录器
    hash_registry = {'NG': {}, 'OK': {}}

    def process_category(source_path, category):
        file_list = []
        for root, _, files in os.walk(source_path):
            for f in files:
                if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                    file_list.append(os.path.join(root, f))

        print(f"\n处理 {category} 图片: {len(file_list)} 张")

        for img_path in tqdm(file_list, desc=f"处理 {category}"):
            filename = os.path.basename(img_path)

            # 1. 损坏检测
            if is_corrupt_image(img_path):
                shutil.copy2(img_path, os.path.join(dest_base, DEST_CORRUPT_NAME, filename))
                counters['corrupt'] += 1
                continue

            # 2. 针线检测 (仅NG类需要)
            if category == "NG" and not detect_needle_presence(img_path):
                shutil.copy2(img_path, os.path.join(dest_base, DEST_NO_NEEDLE_NAME, filename))
                counters['no_needle'] += 1
                continue

            # 3. 感知哈希去重
            img_hash = compute_image_hash(img_path)
            if img_hash is None:
                continue

            is_duplicate = False
            for stored_hash in hash_registry[category]:
                if img_hash - stored_hash <= HASH_THRESHOLD:
                    is_duplicate = True
                    break

            if is_duplicate:
                shutil.copy2(img_path, os.path.join(dest_base, DEST_DUPLICATE_NAME, filename))
                counters['duplicate'] += 1
                continue

            # 4. 针线细节增强 (可选)
            dest_folder = os.path.join(dest_base, dest_ng_name if category == "NG" else dest_ok_name)
            dest_path = os.path.join(dest_folder, filename)

            if enhance_details:
                try:
                    img = cv2.imread(img_path)
                    enhanced = enhance_needle_details(img)
                    cv2.imwrite(dest_path, enhanced)
                    counters['enhanced'] += 1
                except:
                    # 增强失败时保存原图
                    shutil.copy2(img_path, dest_path)
            else:
                shutil.copy2(img_path, dest_path)

            # 记录哈希值
            hash_registry[category][img_hash] = True
            counters['ng' if category == "NG" else 'ok'] += 1

    # 处理NG类别
    ng_source = os.path.join(source_base, source_ng_name)
    if os.path.exists(ng_source):
        process_category(ng_source, "NG")
    else:
        print(f"警告: NG目录不存在 {ng_source}")

    # 处理OK类别
    ok_source = os.path.join(source_base, source_ok_name)
    if os.path.exists(ok_source):
        process_category(ok_source, "OK")
    else:
        print(f"警告: OK目录不存在 {ok_source}")

    # 输出统计结果
    print("\n" + "=" * 40)
    print("数据清洗完成".center(40))
    print("=" * 40)
    print(f"有效NG图片: {counters['ng']}")
    print(f"有效OK图片: {counters['ok']}")
    print(f"损坏图片: {counters['corrupt']} (保存到 {DEST_CORRUPT_NAME})")
    print(f"重复图片: {counters['duplicate']} (保存到 {DEST_DUPLICATE_NAME})")
    print(f"无针线NG图片: {counters['no_needle']} (保存到 {DEST_NO_NEEDLE_NAME})")
    if enhance_details:
        print(f"增强处理图片: {counters['enhanced']}")
    print("=" * 40)


if __name__ == "__main__":
    if not os.path.isdir(SOURCE_BASE_DIR):
        print(f"错误: 源目录不存在 {SOURCE_BASE_DIR}")
    else:
        process_images(
            SOURCE_BASE_DIR,
            SOURCE_NG_FOLDER_NAME,
            SOURCE_OK_FOLDER_NAME,
            DESTINATION_BASE_DIR,
            DEST_NG_FOLDER_NAME,
            DEST_OK_FOLDER_NAME,
            enhance_details=True  # 开启针线增强
        )