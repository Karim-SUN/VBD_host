# precompute_clusters_shape.py
import numpy as np
import faiss
import pickle
import os
from collections import defaultdict

# --- 配置 ---
NUM_CLUSTERS = 128  # 你想要的K个簇
ANCHOR_FILE = "./16384.npy"  # 你的 [8192, 40, 3] 锚点文件
OUTPUT_FILE = "./clusters_shape.pkl" # 输出的聚类信息
# ---

print(f"Loading anchors from {ANCHOR_FILE}...")
anchors_traj = np.load(ANCHOR_FILE)
# 获取 x, y 坐标, 形状 [8192, 40, 2]
anchors_xy = anchors_traj[..., :2]

# 1. *** 关键: 计算位移 (np.diff) ***
# 这将聚类的重点从“绝对位置”转移到“运动形状”
# (8192, 40, 2) -> (8192, 39, 2)
anchors_diffs = np.diff(anchors_xy, axis=1) 
print(f"Calculated diffs shape: {anchors_diffs.shape}")

# 2. 扁平化用于 K-Means
# (8192, 39 * 2) -> (8192, 78)
anchor_diffs_flat = anchors_diffs.reshape(anchors_diffs.shape[0], -1).astype(np.float32)
D_dim = anchor_diffs_flat.shape[1] # D_dim 应该是 78

print(f"Running K-Means (k={NUM_CLUSTERS}) on {anchor_diffs_flat.shape[0]} shape vectors of dim {D_dim}...")

# 3. K-Means
kmeans = faiss.Kmeans(d=D_dim, k=NUM_CLUSTERS, niter=30, verbose=True, gpu=False)
kmeans.train(anchor_diffs_flat)

print("K-Means training complete.")

# 4. 获取簇中心 (形状是 [K, 78])
cluster_centers_flat = kmeans.centroids

# *** 这是修正的地方 ***
# 必须 reshape 回 (K, 39, 2) 
cluster_centers_diffs = cluster_centers_flat.reshape(NUM_CLUSTERS, 39, 2)

# 5. 获取每个锚点的簇标签
D, I = kmeans.index.search(anchor_diffs_flat, k=1)
labels = I.squeeze(-1) # [8192]

# 6. 计算每个簇对应的原始锚点索引
print("Mapping cluster IDs to anchor indices...")
cluster_to_indices = defaultdict(list)
for anchor_index, cluster_id in enumerate(labels):
    cluster_to_indices[cluster_id].append(anchor_index)
cluster_to_indices = dict(cluster_to_indices)
print(f"Found {len(cluster_to_indices)} active clusters.")

# 7. 保存到文件
cluster_data = {
    # 保存的是“运动模式”的中心 [K, 39, 2]
    'cluster_centers_diffs': cluster_centers_diffs, 
    'cluster_to_indices': cluster_to_indices,
    'labels': labels,
    'num_clusters': NUM_CLUSTERS
}

os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
with open(OUTPUT_FILE, 'wb') as f:
    pickle.dump(cluster_data, f)

print(f"Cluster (shape) data saved to {OUTPUT_FILE}")
print("Done.")