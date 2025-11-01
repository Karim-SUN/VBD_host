# visualize_clusters_shape_with_centers.py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle
import os

# --- 配置 ---
# !! 确保 .pkl 和 .npy 是匹配的 !!
# (例如，都是基于 16384 个锚点生成的)
PKL_FILE = "./clusters_nms.pkl"  
ANCHOR_FILE = "./16384.npy"     
OUTPUT_IMAGE_FILE = "./cluster_shape_visualization_with_centers.png"
# ---

print(f"Loading cluster data from {PKL_FILE}...")
if not os.path.exists(PKL_FILE):
    print(f"Error: Cluster file not found at {PKL_FILE}")
    print("Please run the corresponding 'precompute_...' script first.")
    exit()

with open(PKL_FILE, 'rb') as f:
    cluster_data = pickle.load(f)

# 提取 'shape' 聚类所需的数据
labels = cluster_data['labels']
num_clusters = cluster_data['num_clusters']
# *** 新增: 加载 'diffs' 簇中心 ***
# 假设形状为 [K, 39, 2]
cluster_centers_diffs = cluster_data.get('cluster_centers_diffs')

if cluster_centers_diffs is None:
    print("Error: 'cluster_centers_diffs' not found in .pkl file.")
    print("This visualization script is only for K-Means (shape) results.")
    exit()

print(f"Loading original anchors from {ANCHOR_FILE}...")
if not os.path.exists(ANCHOR_FILE):
    print(f"Error: Anchor file not found at {ANCHOR_FILE}")
    exit()

anchors_traj = np.load(ANCHOR_FILE)
anchors_xy = anchors_traj[..., :2] # [N, 40, 2]

# --- 检查数据匹配 ---
N_anchors = anchors_xy.shape[0]
N_labels = labels.shape[0]
if N_anchors != N_labels:
    print(f"--- FATAL ERROR ---")
    print(f"Data Mismatch: Anchors file has {N_anchors} entries.")
    print(f"Pickle file has {N_labels} labels.")
    print(f"These files do not match. Please re-run clustering on {ANCHOR_FILE}.")
    print("-------------------")
    exit()

print("Data loaded. Reconstructing cluster centers using cumsum...")

# --- 关键: 重建簇中心轨迹 ---
# 1. 计算所有轨迹的平均起始点
avg_start_point = np.mean(anchors_xy[:, 0, :], axis=0) # shape [2]

K = num_clusters
diff_len = cluster_centers_diffs.shape[1] # 39
traj_len = diff_len + 1                   # 40

# 2. 准备一个 [K, 40, 2] 的空数组
cluster_centers_traj = np.zeros((K, traj_len, 2), dtype=np.float32)

# 3. 将所有轨迹的第0个点设置为平均起始点
cluster_centers_traj[:, 0, :] = avg_start_point

# 4. 使用 cumsum 计算剩余的 39 个点
# cluster_centers_diffs [K, 39, 2]
# np.cumsum(...)        [K, 39, 2]
cluster_centers_traj[:, 1:, :] = avg_start_point + np.cumsum(cluster_centers_diffs, axis=1)

print("Reconstruction complete. Generating plot...")

# --- 可视化 ---
plt.figure(figsize=(12, 10))
ax = plt.gca()
cmap = plt.get_cmap('gist_rainbow', num_clusters)

# 1. 绘制所有锚点轨迹 (彩色)
print(f"Plotting all {N_anchors} anchor trajectories (colored by cluster)...")
for i in range(N_anchors):
    traj = anchors_xy[i]
    label = labels[i]
    
    # 修复当 num_clusters=1 时的除零错误
    norm_label = (label / (num_clusters - 1)) if num_clusters > 1 else 0.5
    color = cmap(norm_label)
    
    ax.plot(traj[:, 0], traj[:, 1], color=color, alpha=0.05, linewidth=0.5)

# 2. *** 新增: 绘制重建的簇中心 (黑色虚线) ***
print(f"Plotting all {K} reconstructed cluster centers (black dashed)...")
for k in range(K):
    center_traj = cluster_centers_traj[k]
    ax.plot(center_traj[:, 0], center_traj[:, 1], 
            color='black', 
            linewidth=2.5, 
            alpha=0.8, 
            linestyle='--')

# 4. 设置图表属性
ax.set_xlabel("X Coordinate", fontsize=12)
ax.set_ylabel("Y Coordinate", fontsize=12)
ax.set_title(f"Clustering of Trajectory Shapes (k={num_clusters}) with Reconstructed Centers", fontsize=14)
ax.grid(True, linestyle=':', alpha=0.6)
ax.set_aspect('equal', adjustable='box')

# 5. 保存图像
plt.tight_layout()
plt.savefig(OUTPUT_IMAGE_FILE, dpi=150)

print(f"Visualization saved to {OUTPUT_IMAGE_FILE}")
print("Done.")