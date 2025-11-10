# precompute_clusters_direct.py
import numpy as np
import faiss
import pickle
import os
# 导入 defaultdict 以便轻松构建索引列表
from collections import defaultdict

# --- 配置 ---
NUM_CLUSTERS = 128  # 你想要的K个簇
ANCHOR_FILE = "./16384.npy"  # 你的 [8192, 40, 3] 锚点文件
# 建议使用新文件名，以免覆盖基于 'diff' 的旧文件
OUTPUT_FILE = "./clusters_direct.pkl" 
# ---

print(f"Loading anchors from {ANCHOR_FILE}...")
# 假设锚点是 [8192, 40, 3] (x, y, yaw)
anchors_traj = np.load(ANCHOR_FILE)

# 1. 准备数据 (Prepare Data) - NO DIFF
# 我们只使用 x, y 进行聚类，形状 [8192, 40, 2]
anchors_xy = anchors_traj[..., :2]

print(f"Original anchors (x,y) shape: {anchors_xy.shape}") # 应为 (8192, 40, 2)

# 2. 扁平化用于 K-Means
# (8192, 40 * 2) -> (8192, 80)
anchors_flat = anchors_xy.reshape(anchors_xy.shape[0], -1).astype(np.float32)
D_dim = anchors_flat.shape[1] # D_dim 应该是 80

print(f"Running K-Means (k={NUM_CLUSTERS}) on {anchors_flat.shape[0]} vectors of dim {D_dim}...")

# 3. K-Means
kmeans = faiss.Kmeans(d=D_dim, k=NUM_CLUSTERS, niter=30, verbose=True, gpu=False)
kmeans.train(anchors_flat)

print("K-Means training complete.")

# 4. 获取簇中心 (簇)
cluster_centers_flat = kmeans.centroids # [K, 80]
# 恢复回轨迹形状 [K, 40, 2]
cluster_centers = cluster_centers_flat.reshape(NUM_CLUSTERS, 40, 2)

# 5. 获取每个锚点的簇标签
# D 是距离, I 是标签 (索引)
D, I = kmeans.index.search(anchors_flat, k=1)
labels = I.squeeze(-1) # 形状 [8192], 值为 0 到 63

# 6. *** 新要求: 计算每个簇对应的原始锚点索引 ***
print("Mapping cluster IDs to anchor indices...")
# 创建一个字典，其默认值为空列表
cluster_to_indices = defaultdict(list)

# 遍历所有 8192 个锚点
# 'anchor_index' 将是 0, 1, 2, ...
# 'cluster_id' 将是 labels[anchor_index] (例如 5, 12, 0, ...)
for anchor_index, cluster_id in enumerate(labels):
    # 将原始索引 (anchor_index) 添加到其对应簇 (cluster_id) 的列表中
    cluster_to_indices[cluster_id].append(anchor_index)

# (可选) 将 defaultdict 转换回标准 dict 以便保存
cluster_to_indices = dict(cluster_to_indices)
print(f"Found {len(cluster_to_indices)} active clusters.")


# 7. 保存到文件
cluster_data = {
    # 簇中心 (K个簇, 每个形状为 [40, 2])
    'cluster_centers': cluster_centers,
    
    # [新] 簇ID -> 原始索引列表 的映射
    # 例如: {0: [0, 5, 12], 1: [3, 4, 10], ...}
    'cluster_to_indices': cluster_to_indices,
    
    # 每个锚点 (0-8191) 对应的簇ID [8192]
    'labels': labels,
    
    'num_clusters': NUM_CLUSTERS
}

os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
with open(OUTPUT_FILE, 'wb') as f:
    pickle.dump(cluster_data, f)

print(f"Cluster data saved to {OUTPUT_FILE}")
print("Done.")