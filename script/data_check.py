import os, pickle
from tqdm import tqdm

root = "/media/huang/PortableSSD/train_processed/processed"
bad_files = []
for file in tqdm(os.listdir(root)):
    if file.endswith(".pkl"):
        path = os.path.join(root, file)
        try:
            with open(path, "rb") as f:
                pickle.load(f)
        except Exception as e:
            print(f"❌ 读取失败: {path} - {e}")
            bad_files.append(path)

print("坏文件数量:", len(bad_files))
