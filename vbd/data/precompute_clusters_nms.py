# precompute_clusters_nms.py
import numpy as np
import faiss
import pickle
import os
from collections import defaultdict
import time

# --- 配置 ---
ANCHOR_FILE = "./16384.npy"
OUTPUT_FILE = "./clusters_nms.pkl"
NUM_CLUSTERS = 128 # 最终想要的簇数量

# --- “NMS” (子采样) 超参数 ---
# 这是您需要调整的关键参数
# "抑制半径": 任何两个“原型”点之间的最小距离
# (在 78 维标准化空间中，一个 5.0 到 10.0 之间的值是合理的)
NMS_THRESHOLD = 5.0

# ---

# *** 1. 数据加载和预处理 (与之前相同) ***
print(f"Loading anchors from {ANCHOR_FILE}...")
anchors_traj = np.load(ANCHOR_FILE)
anchors_xy = anchors_traj[..., :2]
N_points, traj_len, _ = anchors_xy.shape # 16384, 40
diff_len = traj_len - 1                 # 39

print(f"Calculating {N_points} shape vectors...")
anchors_diffs = np.diff(anchors_xy, axis=1) # [N, 39, 2]
anchor_diffs_flat = anchors_diffs.reshape(N_points, -1).astype(np.float32) # [N, 78]
D_dim = anchor_diffs_flat.shape[1]

# 手动标准化数据
print("Scaling data (manual)...")
mean = np.mean(anchor_diffs_flat, axis=0)
std = np.std(anchor_diffs_flat, axis=0)
std[std == 0] = 1.0
data_scaled = (anchor_diffs_flat - mean) / std
data_scaled = data_scaled.astype(np.float32)
print("Scaling complete.")

# ---
# *** 2. 阶段 1: “NMS” 子采样 ***
# ---
print(f"--- Stage 1: Running NMS Subsampling (Threshold={NMS_THRESHOLD}) ---")
start_time = time.time()

# 随机打乱索引，使选择与顺序无关
indices_shuffled = np.random.permutation(N_points)

# 跟踪哪些点已被抑制
# 使用布尔数组比 `set()` 更快
suppressed = np.zeros(N_points, dtype=bool) 
kept_indices = []

# 为整个数据集构建一个 FAISS 索引
index_nms = faiss.IndexFlatL2(D_dim)
index_nms.add(data_scaled)

# 搜索半径的平方
threshold_sq = NMS_THRESHOLD**2

# 遍历所有 16,384 个点
for i in indices_shuffled:
    if suppressed[i]:
        continue # 这个点已经被抑制，跳过
    
    # *** 这是“原型”，保留它 ***
    kept_indices.append(i)
    suppressed[i] = True # (以防万一，尽管我们不会再访问它)

    # 查找它的邻居
    # (注意: faiss.index.range_search 速度较慢, 
    # 我们用 k_search 找一个近似的大邻域)
    # k_search(query, k) 比 range_search 快得多
    # 我们假设一个大簇中至少有 N 个邻居
    K_NEIGHBORS = 100 # 假设一个大簇至少有100个邻居
    D, I = index_nms.search(data_scaled[i:i+1], K_NEIGHBORS)
    
    # 抑制所有在 NMS_THRESHOLD 范围内的邻居
    # D 包含距离的平方
    neighbors_to_suppress = I[0][D[0] < threshold_sq]
    
    # 使用 NumPy 的布尔索引进行快速抑制
    suppressed[neighbors_to_suppress] = True

end_time = time.time()
N_prototypes = len(kept_indices)
print(f"NMS subsampling complete in {end_time - start_time:.2f}s.")
print(f"Reduced {N_points} points -> {N_prototypes} prototypes.")

# 获取 K-Means 要用的“原型”数据集
data_for_kmeans = data_scaled[kept_indices]

# ---
# *** 2. 阶段 1 (续): 在“原型”上运行 K-Means ***
# ---
print(f"Running K-Means (k={NUM_CLUSTERS}) on {N_prototypes} prototypes...")
kmeans = faiss.Kmeans(d=D_dim, k=NUM_CLUSTERS, niter=50, verbose=True, gpu=False)
kmeans.train(data_for_kmeans)

# 这就是我们高质量的簇中心
cluster_centers_flat_scaled = kmeans.centroids # [K, 78]

print("K-Means on prototypes complete.")

# ---
# *** 3. 阶段 2: 将所有 16,384 个点分配给这 K 个中心 ***
# ---
print(f"--- Stage 2: Assigning all {N_points} points to {NUM_CLUSTERS} centers ---")
# 建立一个只包含 K 个簇中心的索引
index_final = faiss.IndexFlatL2(D_dim)
index_final.add(cluster_centers_flat_scaled)

# 搜索 *全部* 16,384 个点
D_final, I_final = index_final.search(data_scaled, k=1)
labels = I_final.squeeze(-1) # 最终的 [16384] 标签数组

print("Full assignment complete.")

# ---
# *** 4. 保存结果 (与之前相同) ***
# ---
# 还原簇中心 (逆标准化)
cluster_centers_flat = (cluster_centers_flat_scaled * std) + mean
cluster_centers_diffs = cluster_centers_flat.reshape(NUM_CLUSTERS, diff_len, 2)

# 计算索引映射
cluster_to_indices = defaultdict(list)
for anchor_index, cluster_id in enumerate(labels):
    cluster_to_indices[int(cluster_id)].append(anchor_index)
cluster_to_indices = dict(cluster_to_indices)

# 保存
cluster_data = {
    'cluster_centers_diffs': cluster_centers_diffs,
    'labels': labels,
    'cluster_to_indices': cluster_to_indices,
    'num_clusters': NUM_CLUSTERS,
    'nms_threshold': NMS_THRESHOLD,
    'num_prototypes': N_prototypes
}

os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
with open(OUTPUT_FILE, 'wb') as f:
    pickle.dump(cluster_data, f)

print(f"Cluster (NMS K-Means) data saved to {OUTPUT_FILE}")
print("Done.")