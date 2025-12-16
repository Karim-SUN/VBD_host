from vbd import waymax_visualization as visualization
import mediapy
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection # 新增导入

def plot_state(
    current_state,
    log_traj = False,
    traj_preds=None, 
    traj_pred_score=None, 
    past_traj_length = 0,
    dx = 75, 
    center_agent_idx = -1, 
    filename = None, 
    t = None, 
    tick_off = False, 
    return_ax = False,
    img_size = (800,800),
    font_size = 12,
    center_xy = None,
    traj_color = 'r',
    cmap = None, # 新增参数: 渐变色 Colormap 名称 (如 'autumn', 'viridis')
    is_ego = None,
    is_adv = None,
):
    viz_config = visualization.utils.VizConfig()
    fig, ax = visualization.utils.init_fig_ax(viz_config)
    if log_traj:
        traj = current_state.log_trajectory
    else:
        traj = current_state.sim_trajectory
    indices = np.arange(traj.num_objects)
    is_controlled = current_state.object_metadata.is_controlled

    # is_ego = np.where(current_state.object_metadata.is_sdc)[0]

    visualization.plot_trajectory(
        ax, traj, is_controlled, time_idx=current_state.timestep,
        indices=indices, past_traj_length = past_traj_length,
        is_ego = is_ego, is_adv = is_adv, show_colorbar=True
    )  # pytype: disable=wrong-arg-types  # jax-ndarray

    # 2. Plots road graph elements.
    visualization.plot_roadgraph_points(ax, current_state.roadgraph_points, verbose=False)
    visualization.plot_traffic_light_signals_as_points(
        ax, current_state.log_traffic_light, current_state.timestep, verbose=False
    )

    current_xy = traj.xy[:, current_state.timestep, :]
    if center_xy is not None:
        origin_x, origin_y = center_xy
    elif center_agent_idx == -1:
        xy = current_xy[current_state.object_metadata.is_sdc]
        origin_x, origin_y = xy[0, :2]
    else:
        xy = current_xy[center_agent_idx]
        origin_x, origin_y = xy[:2]
    # Zoom
    
    ax.axis((
        origin_x - dx,
        origin_x + dx,
        origin_y - dx,
        origin_y + dx,
    ))
    if t is None:
        t = (current_state.timestep-10)/10
    # if font_size>0:
    #     ax.text(origin_x - 0.9*dx, origin_y + 0.9*dx, f"t={t:.1f} s", fontsize=font_size)
    
    if tick_off:
        plt.tick_params(left = False, right = False , labelleft = False , 
                labelbottom = False, bottom = False) 

    if traj_preds is not None:
        T, D = traj_preds.shape[-2:]
        
        # 统一处理没有 score 的情况，构造全 1 的 score 以简化逻辑
        if traj_pred_score is None:
            # 使用一个虚拟的 score 列表，值为 1.0 (代表不透明度最大)
            # 注意：这里我们不需要真实的 filter，只需占位
            scores_to_iter = np.ones(len(traj_preds.reshape(-1, T, D)))
            score_threshold = -1.0 # 不过滤
        else:
            scores_to_iter = traj_pred_score.reshape(-1)
            score_threshold = 0.01

        # 遍历每条预测轨迹
        for traj, score in zip(traj_preds.reshape(-1, T, D), scores_to_iter):
            if score < score_threshold:
                continue
            
            # 计算 Alpha 值 (如果有 score 则应用 score，否则默认 0.8)
            alpha_val = score * 0.8 + 0.2 if traj_pred_score is not None else 0.8

            if cmap is not None:
                # === 渐变色绘制逻辑 ===
                # 1. 准备点集：(T, 1, 2)
                points = traj[:, :2].reshape(-1, 1, 2)
                # 2. 构造线段：(T-1, 2, 2) -> 连接点 i 和 i+1
                segments = np.concatenate([points[:-1], points[1:]], axis=1)
                
                # 3. 创建 LineCollection
                # norm 控制颜色映射范围 (从时间步 0 到 T-1)
                norm = plt.Normalize(0, T-1)
                lc = LineCollection(segments, cmap=cmap, norm=norm)
                
                # 4. 设置颜色映射依据 (这里使用时间步索引)
                lc.set_array(np.arange(T-1))
                lc.set_linewidth(2.0) # 设置线宽
                lc.set_alpha(alpha_val) # 设置透明度
                
                # 5. 添加到图表
                ax.add_collection(lc)
            else:
                # === 原有单色绘制逻辑 ===
                ax.plot(traj[:, 0], traj[:, 1], color=traj_color, alpha=alpha_val)
        
    fig.subplots_adjust(
        left=0.08, bottom=0.08, right=0.98, top=0.98, wspace=0.0, hspace=0.0
    )
    if filename is not None:
        plt.savefig(filename,
                    bbox_inches='tight', 
                    transparent=False,
                    pad_inches=0.02)
    if return_ax:
        return fig, ax
    return mediapy.resize_image(visualization.utils.img_from_fig(fig), img_size)