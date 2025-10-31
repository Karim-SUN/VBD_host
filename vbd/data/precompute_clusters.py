# precompute_clusters.py
import numpy as np
import faiss
import pickle
import os

# --- 配置 ---
NUM_CLUSTERS = 128  # 你想要的K个簇
ANCHOR_FILE = "vbd/data/8192.npy"  # 你的 [8192, 40, 3] 锚点文件
OUTPUT_FILE = "vbd/data/clusters_K128.pkl" # 输出的聚类信息
# ---

print(f"Loading anchors from {ANCHOR_FILE}...")
# 假设锚点是 [8192, 40, 3] (x, y, yaw)
# 我们只使用 x, y 进行聚类，形状 [8192, 40, 2]
anchors_full = np.load(ANCHOR_FILE)
anchors_xy = anchors_full[..., :2] # [8192, 40, 2]

# 假设 VBD_v2.py 的逻辑是 `::2` 采样后 `diff`
# 我们需要匹配这个逻辑
# 1. 构造 81 步轨迹
#    (这个逻辑假设 8192.npy 是 80 步的... 假设 8192.npy 是 [8192, 81, 2] 轨迹)
#    !!!! 
#    !!!! 假设 `8192.npy` 的形状是 [8192, 81, 2] (x, y 轨迹),
#    !!!! 与 VBD_v2.py 中的 `interpolate_anchors` 保持一致。
#    !!!! 如果你的 `[8192, 40, 3]` 已经是 `diffs`，请直接加载并切片 `[..., :2]`
#    !!!!
#    !!!! (假设 8192.npy 是 [8192, 81, 2] 轨迹)
anchors_traj = np.load(ANCHOR_FILE)
anchors_diffs = np.diff(anchors_traj[:, ::2, :2], axis=1) # [8192, 40, 2]

print(f"Original diffs shape: {anchors_diffs.shape}")

# 2. 扁平化用于 K-Means
# [8192, 80] (40 * 2)
anchor_diffs_flat = anchors_diffs.reshape(anchors_diffs.shape[0], -1).astype(np.float32)
D_dim = anchor_diffs_flat.shape[1] # 应该是 80

print(f"Running K-Means (k={NUM_CLUSTERS}) on {anchor_diffs_flat.shape[0]} vectors of dim {D_dim}...")

# 3. K-Means
kmeans = faiss.Kmeans(d=D_dim, k=NUM_CLUSTERS, niter=30, verbose=True, gpu=False)
kmeans.train(anchor_diffs_flat)

print("K-Means training complete.")

# 4. 获取簇中心
cluster_centers_flat = kmeans.centroids # [K, 80]
cluster_centers_diffs = cluster_centers_flat.reshape(NUM_CLUSTERS, 40, 2) # [K, 40, 2]

# 5. 获取每个锚点的簇标签
D, I = kmeans.index.search(anchor_diffs_flat, k=1)
labels = I.squeeze(-1) # [8192]

# 6. 保存到文件
cluster_data = {
    'cluster_centers_diffs': cluster_centers_diffs,
    'labels': labels,
    'num_clusters': NUM_CLUSTERS
}

os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
with open(OUTPUT_FILE, 'wb') as f:
    pickle.dump(cluster_data, f)

print(f"Cluster data saved to {OUTPUT_FILE}")
print("Done.")