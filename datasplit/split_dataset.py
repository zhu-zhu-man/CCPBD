"""
数据集划分脚本
递归查找源目录中最底层包含 image/newmask/newedge/dist 的目录，
汇总所有样本后按 7:2:1 划分为训练集、验证集和测试集。
"""

import os
import shutil
import random

# 源数据根目录
SOURCE_ROOT = r"source_root"  # 请替换为实际路径，例如 C:\Users\Administrator\Desktop\CCPBD\source_root
# 目标数据根目录
TARGET_ROOT = r"target_root"  # 请替换为实际路径，例如 C:\Users\Administrator\Desktop\CCPBD\target_root

# 源文件夹名称
SOURCE_FOLDERS = {
    'image': 'image',
    'mask': 'mask',
    'edge': 'edge',
    'dist': 'dist'
}

# 划分比例
TRAIN_RATIO = 0.7
VAL_RATIO = 0.2
TEST_RATIO = 0.1

# 随机种子，保证结果可复现
RANDOM_SEED = 42


def get_basename_to_file_map(folder_path):
    """构建 basename -> 文件名 的映射（同名时保留首次出现）"""
    mapping = {}
    for f in os.listdir(folder_path):
        full_path = os.path.join(folder_path, f)
        if os.path.isfile(full_path):
            basename = os.path.splitext(f)[0]
            if basename not in mapping:
                mapping[basename] = f
    return mapping


def find_dataset_roots(source_root, source_folders):
    """递归查找同时包含所有要求子目录的数据根目录"""
    required_subdirs = set(source_folders.values())
    dataset_roots = []

    for dirpath, dirnames, _ in os.walk(source_root):
        if required_subdirs.issubset(set(dirnames)):
            dataset_roots.append(dirpath)

    return dataset_roots


def create_directories(target_root):
    """创建目标文件夹结构"""
    splits = ['train', 'val', 'test']
    sub_folders = ['image', 'mask', 'edge', 'dist']
    
    for split in splits:
        for sub in sub_folders:
            folder_path = os.path.join(target_root, split, sub)
            os.makedirs(folder_path, exist_ok=True)
            print(f"创建文件夹: {folder_path}")


def split_dataset(file_basenames, train_ratio, val_ratio, test_ratio, seed):
    """将文件列表按比例划分"""
    random.seed(seed)
    files = file_basenames.copy()
    random.shuffle(files)
    
    total = len(files)
    train_count = int(total * train_ratio)
    val_count = int(total * val_ratio)
    
    train_files = files[:train_count]
    val_files = files[train_count:train_count + val_count]
    test_files = files[train_count + val_count:]
    
    return train_files, val_files, test_files


def collect_all_samples(source_root, dataset_roots, source_folders):
    """从所有数据根目录汇总样本，仅保留四类文件都存在的 basename"""
    all_samples = []

    for root in dataset_roots:
        folder_maps = {}
        for target_sub, source_sub in source_folders.items():
            folder_path = os.path.join(root, source_sub)
            folder_maps[target_sub] = get_basename_to_file_map(folder_path)

        # 取四类数据共同的 basename
        common_basenames = None
        for file_map in folder_maps.values():
            basename_set = set(file_map.keys())
            if common_basenames is None:
                common_basenames = basename_set
            else:
                common_basenames &= basename_set

        if not common_basenames:
            print(f"警告: 目录 {root} 未找到可配对样本")
            continue

        # 生成样本，输出文件名加上相对路径前缀，避免跨省重名覆盖
        rel_root = os.path.relpath(root, source_root)
        rel_prefix = rel_root.replace('\\', '_').replace('/', '_')

        for basename in sorted(common_basenames):
            files = {}
            for target_sub, source_sub in source_folders.items():
                filename = folder_maps[target_sub][basename]
                files[target_sub] = os.path.join(root, source_sub, filename)

            sample_stem = f"{rel_prefix}__{basename}" if rel_prefix not in ['.', ''] else basename
            all_samples.append({
                'stem': sample_stem,
                'files': files
            })

    return all_samples


def copy_files(samples, target_root, split_name):
    """复制样本到目标文件夹"""
    copied_count = 0

    for sample in samples:
        stem = sample['stem']
        files = sample['files']

        for target_sub, source_path in files.items():
            target_folder = os.path.join(target_root, split_name, target_sub)
            source_file = os.path.basename(source_path)
            ext = os.path.splitext(source_file)[1]
            target_name = f"{stem}{ext}"
            target_path = os.path.join(target_folder, target_name)
            shutil.copy2(source_path, target_path)

        copied_count += 1

    return copied_count


def main():
    print("=" * 60)
    print("数据集划分脚本")
    print("=" * 60)
    print(f"源数据目录: {SOURCE_ROOT}")
    print(f"目标数据目录: {TARGET_ROOT}")
    print(f"划分比例 - 训练:验证:测试 = {TRAIN_RATIO}:{VAL_RATIO}:{TEST_RATIO}")
    print("=" * 60)

    # 递归查找数据根目录（同时包含 images/newlabels/newboundary/dist）
    print("\n扫描源目录，查找各省最底层数据目录...")
    dataset_roots = find_dataset_roots(SOURCE_ROOT, SOURCE_FOLDERS)
    if not dataset_roots:
        print("错误: 未找到满足条件的数据目录。")
        print("请检查 SOURCE_ROOT 是否正确，以及子目录是否包含 images/newlabels/newboundary/dist")
        return

    print(f"找到 {len(dataset_roots)} 个数据目录:")
    for p in dataset_roots:
        print(f"  - {p}")

    # 汇总样本
    print("\n汇总所有样本...")
    all_samples = collect_all_samples(SOURCE_ROOT, dataset_roots, SOURCE_FOLDERS)
    if not all_samples:
        print("错误: 未找到可用样本（四类数据需按 basename 一一对应）。")
        return

    print(f"总共汇总 {len(all_samples)} 个样本")
    
    # 创建目标文件夹结构
    print("\n创建目标文件夹结构...")
    create_directories(TARGET_ROOT)
    
    # 划分数据集
    print("\n划分数据集...")
    train_files, val_files, test_files = split_dataset(
        all_samples, TRAIN_RATIO, VAL_RATIO, TEST_RATIO, RANDOM_SEED
    )

    total_count = len(all_samples)
    print(f"训练集: {len(train_files)} 个样本 ({len(train_files)/total_count*100:.1f}%)")
    print(f"验证集: {len(val_files)} 个样本 ({len(val_files)/total_count*100:.1f}%)")
    print(f"测试集: {len(test_files)} 个样本 ({len(test_files)/total_count*100:.1f}%)")
    
    # 复制文件
    print("\n开始复制文件...")

    print("\n复制训练集...")
    train_copied = copy_files(train_files, TARGET_ROOT, 'train')
    print(f"训练集复制完成: {train_copied} 个样本")

    print("\n复制验证集...")
    val_copied = copy_files(val_files, TARGET_ROOT, 'val')
    print(f"验证集复制完成: {val_copied} 个样本")

    print("\n复制测试集...")
    test_copied = copy_files(test_files, TARGET_ROOT, 'test')
    print(f"测试集复制完成: {test_copied} 个样本")

    print("\n" + "=" * 60)
    print("数据集划分完成!")
    print("=" * 60)
    print(f"目标文件夹结构:")
    print(f"  {TARGET_ROOT}")
    print(f"  ├── train")
    print(f"  │   ├── image ({len(train_files)} 个文件)")
    print(f"  │   ├── mask ({len(train_files)} 个文件)")
    print(f"  │   ├── edge ({len(train_files)} 个文件)")
    print(f"  │   └── dist ({len(train_files)} 个文件)")
    print(f"  ├── val")
    print(f"  │   ├── image ({len(val_files)} 个文件)")
    print(f"  │   ├── mask ({len(val_files)} 个文件)")
    print(f"  │   ├── edge ({len(val_files)} 个文件)")
    print(f"  │   └── dist ({len(val_files)} 个文件)")
    print(f"  └── test")
    print(f"      ├── image ({len(test_files)} 个文件)")
    print(f"      ├── mask ({len(test_files)} 个文件)")
    print(f"      ├── edge ({len(test_files)} 个文件)")
    print(f"      └── dist ({len(test_files)} 个文件)")


if __name__ == "__main__":
    main()
