"""
批量重命名各省目录下最底层数据文件夹。

将以下文件夹名：
- boundary -> edge
- images -> image
- labels -> mask
- newboundary -> newedge
- newlabels -> newmask

用法：
- 直接在代码中修改 ROOT_DIR 和 DRY_RUN
- 然后运行：python rename_folders.py

说明：
- 仅重命名“最底层文件夹”（即该文件夹下不再包含子目录）。
- 若目标名称已存在，则跳过并给出提示，避免覆盖。
"""

import os
from typing import Dict, List, Tuple

RENAME_MAP: Dict[str, str] = {
    "boundary": "edge",
    "images": "image",
    "labels": "mask",
    "newboundary": "newedge",
    "newlabels": "newmask",
}

# ====== 在这里直接配置参数 ======
ROOT_DIR = r"path"
DRY_RUN = False  # True: 仅预演，不实际改名；False: 实际执行


def is_leaf_dir(path: str) -> bool:
    """判断目录是否为最底层目录（不含子目录）。"""
    try:
        for name in os.listdir(path):
            if os.path.isdir(os.path.join(path, name)):
                return False
        return True
    except PermissionError:
        return False


def rename_leaf_dirs(root: str, rename_map: Dict[str, str], dry_run: bool = False) -> Tuple[List[str], List[str]]:
    """
    在 root 下自底向上遍历并重命名最底层目录。

    Returns:
        success_logs: 成功（或预演）日志
        skip_logs: 跳过日志
    """
    success_logs: List[str] = []
    skip_logs: List[str] = []

    # topdown=False 确保先处理深层目录
    for dirpath, dirnames, _ in os.walk(root, topdown=False):
        current_name = os.path.basename(dirpath)

        if current_name not in rename_map:
            continue

        if not is_leaf_dir(dirpath):
            skip_logs.append(f"[跳过-非最底层] {dirpath}")
            continue

        parent = os.path.dirname(dirpath)
        new_name = rename_map[current_name]
        new_path = os.path.join(parent, new_name)

        if os.path.exists(new_path):
            skip_logs.append(f"[跳过-目标已存在] {dirpath} -> {new_path}")
            continue

        if dry_run:
            success_logs.append(f"[预演] {dirpath} -> {new_path}")
        else:
            os.rename(dirpath, new_path)
            success_logs.append(f"[成功] {dirpath} -> {new_path}")

    return success_logs, skip_logs


def main() -> None:
    root = os.path.abspath(ROOT_DIR)
    if not os.path.isdir(root):
        print(f"错误：目录不存在 -> {root}")
        return

    print("=" * 70)
    print("开始批量重命名最底层目录")
    print(f"根目录: {root}")
    print(f"模式: {'预演' if DRY_RUN else '实际执行'}")
    print("=" * 70)

    success_logs, skip_logs = rename_leaf_dirs(root, RENAME_MAP, dry_run=DRY_RUN)

    for line in success_logs:
        print(line)
    for line in skip_logs:
        print(line)

    print("-" * 70)
    print(f"完成。成功/预演: {len(success_logs)}，跳过: {len(skip_logs)}")


if __name__ == "__main__":
    main()
