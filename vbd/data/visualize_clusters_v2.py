# visualize_clusters.py
import numpy as np
import matplotlib
# 设置一个非 GUI 后端以防止在某些环境中出错
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle
import os

# --- 配置 ---
# 这个文件必须是上一步 (precompute_clusters_direct.py) 生成的文件
PKL_FILE = "./clusters_direct.pkl" 
# 这是你的原始锚点文件
ANCHOR_FILE = "./16384.npy"  
OUTPUT_IMAGE_FILE = "./cluster_visualization.png"
# ---

print(f"Loading cluster data from {PKL_FILE}...")
if not os.path.exists(PKL_FILE):
    print(f"Error: Cluster file not found at {PKL_FILE}")
    print("Please run the 'precompute_clusters_direct.py' script first.")
    exit()

with open(PKL_FILE, 'rb') as f:
    cluster_data = pickle.load(f)

# 从文件中提取数据
cluster_centers = cluster_data['cluster_centers']
labels = cluster_data['labels']
num_clusters = cluster_data['num_clusters']

print(f"Loading original anchors from {ANCHOR_FILE}...")
if not os.path.exists(ANCHOR_FILE):
    print(f"Error: Anchor file not found at {ANCHOR_FILE}")
    exit()
    
anchors_traj = np.load(ANCHOR_FILE)
anchors_xy = anchors_traj[..., :2]

print("Data loaded. Generating plot...")

# --- 可视化 ---

plt.figure(figsize=(12, 10))
ax = plt.gca()

# 1. 创建一个颜色映射表
cmap = plt.get_cmap('gist_rainbow', num_clusters)

# 2. 绘制所有 8192 个原始锚点轨迹
# 颜色由它们的簇标签 (labels) 决定
print(f"Plotting all {anchors_xy.shape[0]} anchor trajectories (colored by cluster)...")
for i in range(anchors_xy.shape[0]):
    traj = anchors_xy[i]  # [40, 2]
    label = labels[i]    # 0 到 (K-1)
    
    # 将标签标准化到 [0, 1] 以便 cmap 使用
    color = cmap(label / (num_clusters - 1)) 
    
    # 绘制轨迹 (x 坐标, y 坐标)
    # 使用低 alpha 透明度，因为线会大量重叠
    ax.plot(traj[:, 0], traj[:, 1], color=color, alpha=0.05, linewidth=0.5)


# 3. 绘制簇中心 (Cluster Centers)
print(f"Plotting all {num_clusters} cluster centers (black dashed lines)...")
for k in range(num_clusters):
    center_traj = cluster_centers[k] # [40, 2]
    
    # 用更粗的、不透明的黑线来绘制它们，使其突出
    ax.plot(center_traj[:, 0], center_traj[:, 1], 
            color='black', 
            linewidth=2.5, 
            alpha=1.0, 
            linestyle='--')

# 4. 设置图表属性
ax.set_xlabel("X Coordinate", fontsize=12)
ax.set_ylabel("Y Coordinate", fontsize=12)
ax.set_title(f"K-Means Clustering of Anchor Trajectories (k={num_clusters})", fontsize=14)
ax.grid(True, linestyle=':', alpha=0.6)
ax.set_aspect('equal', adjustable='box')

# 5. 保存图像
plt.tight_layout()
plt.savefig(OUTPUT_IMAGE_FILE, dpi=150)

print(f"Visualization saved to {OUTPUT_IMAGE_FILE}")
print("Done.")