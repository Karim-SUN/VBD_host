# visualize_clusters_shape_with_centers_batched.py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle
import os

# --- 配置 ---
PKL_FILE = "./clusters_nms.pkl"  
ANCHOR_FILE = "./16384.npy"     
OUTPUT_IMAGE_FILE = "./cluster_shape_visualization_with_centers.png"
BATCH_SIZE = 16  # *** 新增: 每张图显示的簇数量 ***
# ---

print(f"Loading cluster data from {PKL_FILE}...")
if not os.path.exists(PKL_FILE):
    print(f"Error: Cluster file not found at {PKL_FILE}")
    exit()

with open(PKL_FILE, 'rb') as f:
    cluster_data = pickle.load(f)

# 提取数据
labels = cluster_data['labels']
num_clusters = cluster_data['num_clusters']
cluster_centers_diffs = cluster_data.get('cluster_centers_diffs')

if cluster_centers_diffs is None:
    print("Error: 'cluster_centers_diffs' not found in .pkl file.")
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
    print(f"--- FATAL ERROR: Data Mismatch ({N_anchors} vs {N_labels}) ---")
    exit()

print("Data loaded. Reconstructing cluster centers...")

# --- 重建簇中心轨迹 ---
avg_start_point = np.mean(anchors_xy[:, 0, :], axis=0) # [2]
K = num_clusters
diff_len = cluster_centers_diffs.shape[1]
traj_len = diff_len + 1

cluster_centers_traj = np.zeros((K, traj_len, 2), dtype=np.float32)
cluster_centers_traj[:, 0, :] = avg_start_point
cluster_centers_traj[:, 1:, :] = avg_start_point + np.cumsum(cluster_centers_diffs, axis=1)

print("Reconstruction complete. Generating batched plots...")

# --- *** 新增: 分组循环绘制 *** ---
# 计算需要多少批次
num_batches = int(np.ceil(K / BATCH_SIZE))
filename_base, filename_ext = os.path.splitext(OUTPUT_IMAGE_FILE)

# 初始化颜色映射，确保跨批次颜色一致
cmap = plt.get_cmap('gist_rainbow', num_clusters)

for batch_idx in range(num_batches):
    # 计算当前批次的簇 ID 范围
    start_k = batch_idx * BATCH_SIZE
    end_k = min((batch_idx + 1) * BATCH_SIZE, K)
    
    print(f"Processing Batch {batch_idx+1}/{num_batches}: Clusters {start_k} to {end_k-1}...")

    # 创建画布
    fig = plt.figure(figsize=(12, 10))
    ax = plt.gca()

    # 1. 筛选并绘制当前批次的锚点 (背景)
    # 使用掩码筛选出属于当前这 16 个簇的原始轨迹
    mask = (labels >= start_k) & (labels < end_k)
    batch_anchors = anchors_xy[mask]
    batch_labels = labels[mask]
    
    # 绘制筛选后的锚点
    for i in range(len(batch_anchors)):
        traj = batch_anchors[i]
        label = batch_labels[i]
        
        # 颜色计算 (保持全局一致)
        norm_label = (label / (num_clusters - 1)) if num_clusters > 1 else 0.5
        color = cmap(norm_label)
        
        ax.plot(traj[:, 0], traj[:, 1], color=color, alpha=0.05, linewidth=0.5)

    # 2. 绘制当前批次的簇中心 (前景)
    for k in range(start_k, end_k):
        center_traj = cluster_centers_traj[k]
        
        # 绘制中心线 (黑色虚线)
        ax.plot(center_traj[:, 0], center_traj[:, 1], 
                color='black', 
                linewidth=2.5, 
                alpha=0.8, 
                linestyle='--')
        
        # 标注编号
        end_x, end_y = center_traj[-1, 0], center_traj[-1, 1]
        ax.text(end_x, end_y, 
                str(k), 
                color='black', 
                fontsize=14,           # 字体调大一点
                fontweight='bold', 
                ha='center', va='center', 
                zorder=10)

    # 设置图表属性
    ax.set_xlabel("X Coordinate", fontsize=12)
    ax.set_ylabel("Y Coordinate", fontsize=12)
    ax.set_title(f"Cluster Shapes (Batch {batch_idx+1}: Clusters {start_k}-{end_k-1})", fontsize=14)
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.set_aspect('equal', adjustable='box')

    # 生成当前批次的文件名
    batch_filename = f"{filename_base}_batch_{batch_idx+1}_{start_k}_{end_k-1}{filename_ext}"
    
    plt.tight_layout()
    plt.savefig(batch_filename, dpi=150)
    plt.close(fig) # 重要: 关闭图形释放内存
    
    print(f"  -> Saved: {batch_filename}")

print("All visualization batches saved.")