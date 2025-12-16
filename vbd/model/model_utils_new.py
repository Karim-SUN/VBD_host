import torch
import numpy as np
from scipy.interpolate import splprep, splev


def batch_transform_trajs_to_local_frame(trajs, ref_idx=-1):
    """
    Batch transform trajectories to the local frame of reference.

    Args:
        trajs (torch.Tensor): Trajectories tensor of shape [B, N, T, x].
        ref_idx (int): Reference index for the local frame. Default is -1.

    Returns:
        torch.Tensor: Transformed trajectories in the local frame.

    """
    x = trajs[..., 0]
    y = trajs[..., 1]
    theta = trajs[..., 2]
    v_x = trajs[..., 3]
    v_y = trajs[..., 4]

    local_x = (x - x[:, :, ref_idx, None]) * torch.cos(theta[:, :, ref_idx, None]) + \
              (y - y[:, :, ref_idx, None]) * torch.sin(theta[:, :, ref_idx, None])
    local_y = -(x - x[:, :, ref_idx, None]) * torch.sin(theta[:, :, ref_idx, None]) + \
              (y - y[:, :, ref_idx, None]) * torch.cos(theta[:, :, ref_idx, None])

    local_theta = theta - theta[:, :, ref_idx, None]
    local_theta = wrap_angle(local_theta)

    local_v_x = v_x * torch.cos(theta[:, :, ref_idx, None]) + v_y * torch.sin(theta[:, :, ref_idx, None])
    local_v_y = -v_x * torch.sin(theta[:, :, ref_idx, None]) + v_y * torch.cos(theta[:, :, ref_idx, None])

    local_trajs = torch.stack([local_x, local_y, local_theta, local_v_x, local_v_y], dim=-1)
    local_trajs[trajs[..., :5] == 0] = 0

    if trajs.shape[-1] > 5:
        trajs = torch.cat([local_trajs, trajs[..., 5:]], dim=-1)
    else:
        trajs = local_trajs

    return trajs


def batch_transform_polylines_to_local_frame(polylines):
    """
    Batch transform polylines to the local frame of reference.

    Args:
        polylines (torch.Tensor): Polylines tensor of shape [B, M, W, 5].

    Returns:
        torch.Tensor: Transformed polylines in the local frame.

    """
    x = polylines[..., 0]
    y = polylines[..., 1]
    theta = polylines[..., 2]

    local_x = (x - x[:, :, 0, None]) * torch.cos(theta[:, :, 0, None]) + \
              (y - y[:, :, 0, None]) * torch.sin(theta[:, :, 0, None])
    local_y = -(x - x[:, :, 0, None]) * torch.sin(theta[:, :, 0, None]) + \
              (y - y[:, :, 0, None]) * torch.cos(theta[:, :, 0, None])

    local_theta = theta - theta[:, :, 0, None]
    local_theta = wrap_angle(local_theta)

    local_polylines = torch.stack([local_x, local_y, local_theta], dim=-1)
    local_polylines[polylines[..., :3] == 0] = 0
    polylines = torch.cat([local_polylines, polylines[..., 3:]], dim=-1)

    return polylines


def batch_transform_trajs_to_global_frame(trajs, current_states):
    """
    Batch transform trajectories to the global frame of reference.

    Args:
        trajs (torch.Tensor): Trajectories tensor of shape [B, N, ..., 2 or 3].
        current_states (torch.Tensor): Current states tensor of shape [B, N, ..., 3].
                                      The dimensions after B, N must be broadcastable to trajs.

    Returns:
        torch.Tensor: Transformed trajectories in the global frame. [B, N, ..., 3]

    """
    x, y, theta = current_states[..., 0], current_states[..., 1], current_states[..., 2]

    # Ensure theta has the correct number of dimensions for broadcasting
    # Add dimensions to theta until it matches the number of dimensions of trajs
    while len(theta.shape) < len(trajs.shape) -1:
        theta = theta.unsqueeze(-1)

    g_x = trajs[..., 0] * torch.cos(theta) - trajs[..., 1] * torch.sin(theta)
    g_y = trajs[..., 0] * torch.sin(theta) + trajs[..., 1] * torch.cos(theta)
    
    # Add dimensions to x and y for broadcasting
    while len(x.shape) < len(g_x.shape):
        x = x.unsqueeze(-1)
        y = y.unsqueeze(-1)

    x = g_x + x
    y = g_y + y

    if trajs.shape[-1] == 2:
        trajs = torch.stack([x, y], dim=-1)
    else:
        g_theta = trajs[..., 2] + theta
        g_theta = wrap_angle(g_theta)
        trajs = torch.stack([x, y, g_theta], dim=-1)

    return trajs


def wrap_angle(angle):
    """
    Wrap the angle to [-pi, pi].

    Args:
        angle (torch.Tensor): Angle tensor.

    Returns:
        torch.Tensor: Wrapped angle.

    """
    # return torch.atan2(torch.sin(angle), torch.cos(angle))
    return (angle + torch.pi) % (2 * torch.pi) - torch.pi


def inverse_kinematics(
        agents_future: torch.Tensor,
        agents_future_valid: torch.Tensor,
        dt: float = 0.1,
        action_len: int = 5,
):
    """
    Perform inverse kinematics to compute actions.

    Args:
        agents_future (torch.Tensor): Future agent positions tensor.
            [B, A, T, 8] # x, y, yaw, velx, vely, length, width, height
        agents_future_valid (torch.Tensor): Future agent validity tensor. [B, A, T]
        dt (float): Time interval. Default is 0.1.
        action_len (int): Length of each action. Default is 5.

    Returns:
        torch.Tensor: Predicted actions.

    """
    # Inverse kinematics implementation goes here
    batch_size, num_agents, num_timesteps, _ = agents_future.shape
    assert (num_timesteps - 1) % action_len == 0, "future_len must be divisible by action_len"
    num_actions = (num_timesteps - 1) // action_len

    skip = int(dt / 0.1)
    yaw = agents_future[..., ::skip, 2]
    speed = torch.norm(agents_future[..., ::skip, 3:5], dim=-1)

    yaw_rate = wrap_angle(torch.diff(yaw, dim=-1)) / dt
    accel = torch.diff(speed, dim=-1) / dt
    action_valid = agents_future_valid[..., :1] & agents_future_valid[..., :-1] & agents_future_valid[..., 1:]
    action_valid = action_valid[..., ::skip]

    # filter out invalid actions
    yaw_rate = torch.where(action_valid, yaw_rate, 0.0)
    accel = torch.where(action_valid, accel, 0.0)

    # Reshape for mean pooling
    yaw_rate = yaw_rate.reshape(batch_size, num_agents, num_actions, -1)
    accel = accel.reshape(batch_size, num_agents, num_actions, -1)
    action_valid = action_valid.reshape(batch_size, num_agents, num_actions, -1)

    yaw_rate_sample = yaw_rate.sum(dim=-1) / torch.clamp(action_valid.sum(dim=-1), min=1.0)
    accel_sample = accel.sum(dim=-1) / torch.clamp(action_valid.sum(dim=-1), min=1.0)
    action = torch.stack([accel_sample, yaw_rate_sample], dim=-1)
    action_valid = action_valid.any(dim=-1)

    # Filter again
    action = torch.where(action_valid[..., None], action, 0.0)

    return action, action_valid


def roll_out(
        current_states: torch.Tensor,
        actions: torch.Tensor,
        dt: float = 0.1,
        action_len: int = 5,
        global_frame: float = True
):
    """
    Forward pass of the dynamics model.

    Args:
        current_states (torch.Tensor): Current states tensor of shape [B, N, x, 5]. [x, y, theta, v_x, v_y]
        actions (torch.Tensor): Inputs tensor of shape [B, N, x, T_f//T_a, 2]. [Accel, yaw_rate]
        global_frame (bool): Flag indicating whether to use the global frame of reference. Default is False.

    Returns:
        torch.Tensor: Predicted trajectories.

    """
    x = current_states[..., 0]
    y = current_states[..., 1]
    theta = current_states[..., 2]
    v_x = current_states[..., 3]
    v_y = current_states[..., 4]
    v = torch.sqrt(v_x ** 2 + v_y ** 2)

    a = actions[..., 0].repeat_interleave(action_len, dim=-1)
    v = v.unsqueeze(-1) + torch.cumsum(a * dt, dim=-1)
    v += torch.clamp(torch.randn_like(v), min=-2, max=2) * 0.05
    v = torch.clamp(v, min=0)

    yaw_rate = actions[..., 1].repeat_interleave(action_len, dim=-1)
    yaw_rate += torch.clamp(torch.randn_like(yaw_rate), min=-2, max=2) * 0.01

    if global_frame:
        theta = theta.unsqueeze(-1) + torch.cumsum(yaw_rate * dt, dim=-1)
    else:
        theta = torch.cumsum(yaw_rate * dt, dim=2)

    # theta = torch.fmod(theta + torch.pi, 2*torch.pi) - torch.pi
    # theta = wrap_angle(theta)

    v_x = v * torch.cos(theta)
    v_y = v * torch.sin(theta)

    if global_frame:
        x = x.unsqueeze(-1) + torch.cumsum(v_x * dt, dim=-1)
        y = y.unsqueeze(-1) + torch.cumsum(v_y * dt, dim=-1)
    else:
        x = torch.cumsum(v_x * dt, dim=-1)
        y = torch.cumsum(v_y * dt, dim=-1)

    return torch.stack([x, y, theta, v_x, v_y], dim=-1)


def roll_out_new_old(
    current_states: torch.Tensor,
    traj_increments: torch.Tensor,
    global_frame: float = True,
    action_len: int = 5
):
    """
    Forward pass of the dynamics model.

    Args:
    current_states (torch.Tensor): Current states tensor of shape [B, N, x, 5]. [x, y, theta, v_x, v_y]
    traj_increments (torch.Tensor): Inputs tensor of shape [B, N, x, T_f//T_a, 2]. [delta_x, delta_y]
    global_frame (bool): Flag indicating whether to use the global frame of reference. Default is False.
    action_len (int): Length of each action increment.

    Returns:
    torch.Tensor: Predicted trajectories.

    """
    x0 = current_states[..., 0]
    y0 = current_states[..., 1]
    theta0 = current_states[..., 2]
    vx0 = current_states[..., 3]
    vy0 = current_states[..., 4]

    B, N, T, D = traj_increments.shape
    
    # Repeat increments to match the desired rollout length
    traj_increments_expanded = traj_increments.repeat_interleave(action_len, dim=-2)
    
    # Calculate cumulative sum for the trajectory points
    traj_cumsum = torch.cumsum(traj_increments_expanded, dim=-2)
    
    # The original trajectory is now the one with cumulative sums
    x_origin = traj_cumsum[..., 0]
    y_origin = traj_cumsum[..., 1]

    # For now, theta, vx, vy are placeholders. This might need a more sophisticated calculation.
    T_origin = traj_cumsum.shape[-2]
    theta_origin = torch.zeros(B, N, T_origin, device=traj_increments.device)
    vx_origin = torch.zeros(B, N, T_origin, device=traj_increments.device)
    vy_origin = torch.zeros(B, N, T_origin, device=traj_increments.device)

    if global_frame:
        x_origin = x0.unsqueeze(-1) + x_origin
        y_origin = y0.unsqueeze(-1) + y_origin
        theta_origin = theta0.unsqueeze(-1) + theta_origin
        vx_origin = vx0.unsqueeze(-1) + vx_origin
        vy_origin = vy0.unsqueeze(-1) + vy_origin

    # The interpolated trajectory is just the original trajectory in this version
    # as interpolation logic is removed for simplification with action_len.
    # You can add interpolation logic here if needed.
    interpolated_traj = torch.stack([x_origin, y_origin, theta_origin, vx_origin, vy_origin], dim=-1)
    original_traj = torch.stack([x_origin, y_origin, theta_origin, vx_origin, vy_origin], dim=-1)

    return interpolated_traj, original_traj


def roll_out_new(
    current_states: torch.Tensor,
    traj_increments: torch.Tensor,
    action_len: int = 5
):
    """
    根据轨迹增量和初始状态，推演局部轨迹和全局轨迹。
    
    Args:
        current_states (torch.Tensor): 当前(初始)状态 [B, N, 5] 或 [B, N, 1, 5]。
                                    格式: [x, y, theta, v_x, v_y]
        traj_increments (torch.Tensor): 轨迹增量 (通常是局部坐标系下) [B, N, K, 2]。
                                        格式: [delta_x, delta_y]
        action_len (int): 每个增量持续的时间步长 (用于上采样)。

    Returns:
        local_traj (torch.Tensor): 局部坐标系下的完整轨迹 [B, N, T, 5]
        global_traj (torch.Tensor): 全局坐标系下的完整轨迹 [B, N, T, 5]
    """
    
    # 1. 解析初始状态
    # 确保维度匹配，如果是 [B, N, 1, 5] 则去掉中间的维度
    if current_states.dim() == 4 and current_states.shape[2] == 1:
        current_states = current_states.squeeze(2)
        
    x0 = current_states[..., 0]      # [B, N]
    y0 = current_states[..., 1]      # [B, N]
    theta0 = current_states[..., 2]  # [B, N]
    vx0 = current_states[..., 3]     # [B, N]
    vy0 = current_states[..., 4]     # [B, N]

    # 2. 处理增量 (Upsampling / Repeat)
    # [B, N, K, 2] -> [B, N, K * action_len, 2]
    # 这里假设 traj_increments 是每一步的位移 (delta_position)
    traj_increments_expanded = traj_increments.repeat_interleave(action_len, dim=-2)
    
    # 3. 构建局部轨迹 (Local Trajectory)
    # 在局部坐标系下，起始点通常认为是 (0, 0)，方向朝向 x轴正方向
    local_x = torch.cumsum(traj_increments_expanded[..., 0], dim=-1) # [B, N, T]
    local_y = torch.cumsum(traj_increments_expanded[..., 1], dim=-1) # [B, N, T]
    
    # 简单估算局部航向角 (通过 arctan2 计算位移向量的角度)
    # 注意：这只是基于位移的瞬时航向，更严谨的做法是如果有 angular_rate 输出则积分
    local_theta = torch.atan2(traj_increments_expanded[..., 1], traj_increments_expanded[..., 0])
    
    # 局部速度 (这里暂用增量大小表示速度，假设 dt=0.1，如果 dt!=0.1 需要除以 dt)
    local_vx = traj_increments_expanded[..., 0] / 0.1
    local_vy = traj_increments_expanded[..., 1] / 0.1

    # 堆叠局部轨迹 [B, N, T, 5]
    local_traj = torch.stack([local_x, local_y, local_theta, local_vx, local_vy], dim=-1)

    # 4. 构建全局轨迹 (Global Trajectory) - 核心修正部分
    # 需要将局部坐标 (local_x, local_y) 旋转 theta0 角度，然后平移 (x0, y0)
    
    # 准备旋转矩阵所需的 cos 和 sin，扩展维度以匹配时间 T
    # theta0: [B, N] -> [B, N, 1]
    c = torch.cos(theta0).unsqueeze(-1)
    s = torch.sin(theta0).unsqueeze(-1)
    
    # 旋转公式 (2D 旋转矩阵):
    # x_global = x_local * cos(theta) - y_local * sin(theta)
    # y_global = x_local * sin(theta) + y_local * cos(theta)
    
    rotated_x = local_x * c - local_y * s
    rotated_y = local_x * s + local_y * c
    
    # 平移
    global_x = x0.unsqueeze(-1) + rotated_x
    global_y = y0.unsqueeze(-1) + rotated_y
    
    # 航向角也需要加上初始角度
    global_theta = local_theta + theta0.unsqueeze(-1)
    # 归一化角度到 [-pi, pi]
    global_theta = torch.atan2(torch.sin(global_theta), torch.cos(global_theta))
    
    # 速度向量也需要旋转 (将局部速度投影到全局坐标系)
    global_vx = local_vx * c - local_vy * s
    global_vy = local_vx * s + local_vy * c
    
    # 堆叠全局轨迹 [B, N, T, 5]
    global_traj = torch.stack([global_x, global_y, global_theta, global_vx, global_vy], dim=-1)

    return local_traj, global_traj


class TrajectoryType:
    STATIONARY = 0
    STRAIGHT = 1
    STRAIGHT_RIGHT = 2
    STRAIGHT_LEFT = 3
    STRAIGHT_LEFT_CHANGE_LANE = 4
    STRAIGHT_RIGHT_CHANGE_LANE = 5
    RIGHT_U_TURN = 6
    RIGHT_TURN = 7
    LEFT_U_TURN = 8
    LEFT_TURN = 9


class Params:
    # Parameters for classification, taken from WOD
    SpeedForStationary = 2.0  # (m/s)
    DisplacementForStationary = 3.0  # (m)  $$$ 5.0
    LateralDisplacementForStraight = 2.5  # (m)  $$$ 5.0
    LongitudinalDisplacementForUTurn = 0.0  # (m)  $$$ -5.0
    AbsHeadingDiffForStraight = np.pi / 4.0  # (rad)  $$$ np.pi / 6.0


def get_trajectory_type(traj: torch.Tensor):
    """
    Get the trajectory type.
    Args:
        traj (torch.Tensor): Trajectory tensor of shape [B*A, T, 3].

    Returns:
        traj_type (torch.Tensor): Trajectory type.
    """

    T = traj.shape[1]

    traj_end_point = traj[:, -1, :2]
    traj_end_heading = traj[:, -1, 2]
    traj_start_point = traj[:, 0, :2]
    traj_start_heading = traj[:, 0, 2]

    # Compute the displacement
    displacement = torch.norm(traj_end_point - traj_start_point, dim=-1)

    # Compute the heading difference
    heading_diff = wrap_angle(traj_end_heading - traj_start_heading)

    # Compute the lateral displacement
    lateral_displacement = torch.norm(traj_end_point - traj_start_point, dim=-1) * torch.sin(heading_diff)

    # Compute the longitudinal displacement
    longitudinal_displacement = torch.norm(traj_end_point - traj_start_point, dim=-1) * torch.cos(heading_diff)

    # Compute nonmonotonicity
    traj_dhead = torch.diff(traj[:, :, -1], dim=-1)
    kernal_size = 5
    kernal = torch.ones(1, kernal_size).cuda() / kernal_size
    pad = (kernal_size - 1) // 2
    traj_dhead = torch.nn.functional.pad(traj_dhead, (pad, pad), mode='replicate')
    traj_dhead_filtered = torch.nn.functional.conv1d(traj_dhead.unsqueeze(1), kernal.unsqueeze(1), groups=1).squeeze(1)
    traj_direction = torch.sign(traj_dhead_filtered)
    traj_reversals = torch.sum((traj_direction[:, :-1] != 0) & (traj_direction[:, :-1] != traj_direction[:, 1:]),
                               dim=-1)
    maxHeadingThreshold = Params.AbsHeadingDiffForStraight / T
    traj_direction_filtered = torch.where(torch.abs(traj_dhead_filtered) < maxHeadingThreshold, 0, traj_direction)
    non_zero_mask = traj_direction_filtered != 0
    traj_direction_first_change, _ = torch.max(traj_direction_filtered * non_zero_mask, dim=1)

    # Compute the trajectory type
    traj_types = torch.zeros_like(displacement, dtype=torch.long)
    for i in range(traj_types.shape[0]):
        if displacement[i] < Params.DisplacementForStationary:
            traj_types[i] = TrajectoryType.STATIONARY
            continue
        if torch.abs(heading_diff[i] < Params.AbsHeadingDiffForStraight):
            if lateral_displacement[i] < Params.LateralDisplacementForStraight:
                traj_types[i] = TrajectoryType.STRAIGHT
                continue
            if traj_reversals[i] >= 2:
                traj_types[i] = TrajectoryType.STRAIGHT_LEFT_CHANGE_LANE \
                    if traj_direction_first_change[i] < 0 else TrajectoryType.STRAIGHT_RIGHT_CHANGE_LANE
                continue
            traj_types[i] = TrajectoryType.STRAIGHT_RIGHT \
                if lateral_displacement[i] < 0 else TrajectoryType.STRAIGHT_LEFT
            continue
        if heading_diff[i] < -Params.AbsHeadingDiffForStraight and lateral_displacement[i] < 0:
            traj_types[i] = TrajectoryType.RIGHT_U_TURN \
                if longitudinal_displacement[i] < Params.LongitudinalDisplacementForUTurn else TrajectoryType.RIGHT_TURN
            continue
        if longitudinal_displacement[i] < Params.LongitudinalDisplacementForUTurn:
            traj_types[i] = TrajectoryType.LEFT_U_TURN
            continue
        traj_types[i] = TrajectoryType.LEFT_TURN

    return traj_types


def get_random_mask(B, A, T, T0, reactive_agents_num=4, task_probabilities=None):
    """
    Get a random mask.

    Args:
        B (int): Batch size.
        A (int): Number of agents.
        T (int): Number of timesteps.
        T0 (int): Length of historical timesteps.
        reactive_agents_num (int): Number of reactive agents for the corresponding task.
        task_probabilities (dict): Probabilities for different tasks.

    Returns:
        torch.Tensor: Random mask.
        torch.Tensor: Random type.
    """
    if task_probabilities is None:
        task_probabilities = {
            'prediction': 0.8,
            'goal_conditioned': 0.08,
            'agent_reactive': 0.08,
            'reconstruction': 0.04
        }

    p_pred = task_probabilities['prediction']
    p_goal = p_pred + task_probabilities['goal_conditioned']
    p_react = p_goal + task_probabilities['agent_reactive']

    mask = torch.ones((B, A, T), dtype=torch.bool)
    type = torch.zeros((B, A), dtype=torch.long).to("cuda")
    for i in range(B):
        p = torch.rand(1).item()
        if p < p_pred:
            # Prediction
            mask[i, :, :T0] = 0
            type[i, :] = 1
        elif p < p_goal:
            # Goal Conditioned
            mask[i, :, :T0] = 0
            mask[i, :, -1] = 0
            type[i, :] = 2
        elif p < p_react:
            # Agent Reactive
            num_selected_agents = min(reactive_agents_num, A)
            selected_agents = torch.randperm(A)[:num_selected_agents]
            mask[i, :, :T0] = 0
            mask[i, selected_agents, :] = 0
            type[i, :] = 3
        else:
            # Reconstruction
            mask[i, :, :] = 0
            type[i, :] = 4
            continue

    return mask, type


def interpolate_anchors(anchors, num_interpolated_points=60):
    """
    对 anchor 点进行插值，生成平滑的轨迹。

    参数:
        anchors (np.ndarray): 输入的 anchor 点，形状为 [N, M, 2]，其中 N 是轨迹数量，M 是原始点数。
        num_interpolated_points (int): 插值后的点数，默认为 60。

    返回:
        np.ndarray: 插值后的轨迹，形状为 [N, K, 2]，其中 K 是插值后的点数。
    """
    N, M, D = anchors.shape
    zero_padding = np.zeros((N, 1, D))
    anchors = np.concatenate([zero_padding, anchors], axis=1)
    interpolated_anchors = np.zeros((N, num_interpolated_points, D))

    for i in range(N):
        # 提取当前轨迹的 x 和 y 坐标
        x = anchors[i, :, 0]
        y = anchors[i, :, 1]

        # 使用样条插值
        tck, u = splprep([x, y], s=0)  # s=0 表示平滑插值
        u_new = np.linspace(0, 1, num_interpolated_points)  # 生成新的参数化点
        x_new, y_new = splev(u_new, tck)

        # 将插值结果保存
        interpolated_anchors[i, :, 0] = x_new
        interpolated_anchors[i, :, 1] = y_new

    interpolated_anchors[:, 0] = 0.0

    return interpolated_anchors
