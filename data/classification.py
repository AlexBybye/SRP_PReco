# import os
# import shutil
# from sklearn.model_selection import train_test_split
#
# def split_data(dataset_dir, train_dir, val_dir, test_size=0.2, random_state=42):
#     # 如果训练集和验证集目录已存在，则删除
#     if os.path.exists(train_dir):
#         shutil.rmtree(train_dir)
#     if os.path.exists(val_dir):
#         shutil.rmtree(val_dir)
#
#     # 创建训练集和验证集目录
#     os.makedirs(train_dir)
#     os.makedirs(val_dir)
#
#     # 处理 NG 目录下的子目录（失败类别）
#     ng_dir = os.path.join(dataset_dir, 'NG')
#     if os.path.exists(ng_dir):
#         for ng_sub_dir in os.listdir(ng_dir):
#             ng_sub_path = os.path.join(ng_dir, ng_sub_dir)
#             if os.path.isdir(ng_sub_path):
#                 # 获取 NG 子目录中的所有图片文件
#                 images = [os.path.join(ng_sub_path, img) for img in os.listdir(ng_sub_path) if img.endswith('.jpg')]
#                 # 划分训练集和验证集
#                 train_imgs, val_imgs = train_test_split(images, test_size=test_size, random_state=random_state)
#                 # 创建训练集和验证集的对应子目录
#                 train_ng_dir = os.path.join(train_dir, 'NG', ng_sub_dir)
#                 val_ng_dir = os.path.join(val_dir, 'NG', ng_sub_dir)
#                 os.makedirs(train_ng_dir)
#                 os.makedirs(val_ng_dir)
#                 # 将图片复制到训练集和验证集目录
#                 for img in train_imgs:
#                     shutil.copy(img, os.path.join(train_ng_dir, os.path.basename(img)))
#                 for img in val_imgs:
#                     shutil.copy(img, os.path.join(val_ng_dir, os.path.basename(img)))
#
#     # 处理 OK 目录（成功类别）
#     ok_dir = os.path.join(dataset_dir, 'OK')
#     if os.path.exists(ok_dir):
#         # 获取 OK 目录中的所有图片文件
#         ok_images = [os.path.join(ok_dir, img) for img in os.listdir(ok_dir) if img.endswith('.jpg')]
#         # 划分训练集和验证集
#         ok_train_imgs, ok_val_imgs = train_test_split(ok_images, test_size=test_size, random_state=random_state)
#         # 创建训练集和验证集的 OK 目录
#         train_ok_dir = os.path.join(train_dir, 'OK')
#         val_ok_dir = os.path.join(val_dir, 'OK')
#         os.makedirs(train_ok_dir)
#         os.makedirs(val_ok_dir)
#         # 将图片复制到训练集和验证集目录
#         for img in ok_train_imgs:
#             shutil.copy(img, os.path.join(train_ok_dir, os.path.basename(img)))
#         for img in ok_val_imgs:
#             shutil.copy(img, os.path.join(val_ok_dir, os.path.basename(img)))
#
#     print(f"训练集已保存到 {train_dir}")
#     print(f"验证集已保存到 {val_dir}")
#
# if __name__ == "__main__":
#     dataset_dir = 'data/my_dataset'  # 原始数据集目录
#     train_dir = 'data/my_dataset/train_dataset'  # 划分后的训练集目录
#     val_dir = 'data/my_dataset/val'  # 划分后的验证集目录
#     split_data(dataset_dir, train_dir, val_dir)