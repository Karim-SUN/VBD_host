import numpy as np
import torch
import bitsandbytes.optim as bnb_optim
import lightning.pytorch as pl
from .modules_offset import Encoder, Denoiser, IntentConditioner
from .utils import DDPM_Sampler
from .model_utils_new import (inverse_kinematics, roll_out, 
                              batch_transform_trajs_to_global_frame,
                              batch_transform_trajs_to_local_frame,
                              get_trajectory_type, interpolate_anchors, roll_out_new)
from torch.nn.functional import smooth_l1_loss, cross_entropy, gumbel_softmax, binary_cross_entropy_with_logits
import math
import pickle
import faiss


class VBD(pl.LightningModule):
    """
    Versatile Behavior Diffusion model.
    """

    def __init__(
            self,
            cfg: dict,
    ):
        """
        Initialize the VBD model.

        Args:
            cfg (dict): Configuration parameters for the model.
        """
        super().__init__()
        self.save_hyperparameters()

        self._dt = 0.1

        self.cfg = cfg
        self._future_len = cfg['future_len']
        self._action_skip = int(self._dt / 0.1)
        self._future_action_len = self._future_len / self._action_skip
        self._agents_len = cfg['agents_len']
        self._action_len = cfg['action_len']
        self._diffusion_steps = cfg['diffusion_steps']
        self._encoder_layers = cfg['encoder_layers']
        self._encoder_version = cfg.get('encoder_version', 'v1')
        self._action_mean = cfg['action_mean']
        self._action_std = cfg['action_std']
        self._random_target = cfg.get('random_target', 0.1)
        self._intent_dropout = cfg.get('intent_dropout', 0.0)
        self._task_probabilities = cfg.get('task_probabilities', None)
        self.anchor_incre_min = cfg['anchor_incre_min']
        self.anchor_incre_max = cfg['anchor_incre_max']
        self._balanced_input = cfg.get('balanced_input', 'prob')
        self._weight_start = cfg.get('weight_start', 1.0)
        self._weight_end = cfg.get('weight_end', 0.0)
        self._weight_anneal_steps = cfg.get('weight_anneal_steps', 10000)

        self._offset_std = cfg.get('offset_std', 2.0)

        self.score_loss_type = cfg.get('score_loss_type', 'bce')
        self.bce_loss_weight = cfg.get('bce_loss_weight', 0.5)

        self.use_dynamic_rank_weight = cfg.get('use_dynamic_rank_weight', False)
        self.rank_weight_start = cfg.get('rank_weight_start', 0.0)
        self.rank_weight_end = cfg.get('rank_weight_end', 0.5)
        self.rank_weight_anneal_steps = cfg.get('rank_weight_anneal_steps', 20000)
        self.use_focused_rank_loss = cfg.get('use_focused_rank_loss', False)
        self.focused_rank_topk = cfg.get('focused_rank_topk', 3)
        self.rank_loss_weight = 0.0
        self.use_hinge_loss = cfg.get('use_hinge_loss', False)
        self.soft_hinge_loss_weight = cfg.get('soft_hinge_loss_weight', 0.0)

        self.goal_loss_weight = cfg.get('goal_loss_weight', 1.0)
        self.score_loss_weight = cfg.get('score_loss_weight', 1.0)
        self.predictor_loss_weight = cfg.get('predictor_loss_weight', 1.0)
        self.traj_loss_weight = cfg.get('traj_loss_weight', 1.0)
        self.denoise_loss_weight = cfg.get('denoise_loss_weight', 1.0)
        self.denoiser_loss_weight = cfg.get('denoiser_loss_weight', 1.0)

        self._train_encoder = cfg.get('train_encoder', True)
        self._train_denoiser = cfg.get('train_denoiser', True)
        self._train_predictor = cfg.get('train_predictor', True)
        self._with_predictor = cfg.get('with_predictor', True)
        self._prediction_type = cfg.get('prediction_type', 'sample')
        self._schedule_type = cfg.get('schedule_type', 'cosine')
        self._replay_buffer = cfg.get('replay_buffer', False)
        self._embeding_dim = cfg.get('embeding_dim', 2)  # By default, the embed is the noised trajectory so the dimension is 5
        self._normalize_anchors = cfg.get('normalize_anchors', False)

        self.batch_size = cfg['batch_size']
        self.accumulate_grad_batches = cfg.get('accumulate_grad_batches', 1)

        self.anchor_path = cfg.get('anchor_path')
        self.cluster_path = cfg.get('cluster_path')
        self.process_anchors()

        self.encoder = Encoder(
            self._encoder_layers,
            version=self._encoder_version,
            history_dropout=cfg.get('history_dropout', 0.2),
            task_probabilities=self._task_probabilities
        )

        self.denoiser = Denoiser(
            future_len=self._future_len,
            action_len=self._action_len,
            agents_len=self._agents_len,
            steps=self._diffusion_steps,
            input_dim=self._embeding_dim,
        )
        if self._with_predictor:
            self.predictor = IntentConditioner(
                future_len=self._future_len,
                agents_len=self._agents_len,
                action_len=self._action_len,
                num_clusters=self.num_clusters,
            )
        else:
            self.predictor = None
            self._train_predictor = False

        self.noise_scheduler = DDPM_Sampler(
            steps=self._diffusion_steps,
            schedule=self._schedule_type,
            s=cfg.get('schedule_s', 0.0),
            e=cfg.get('schedule_e', 1.0),
            tau=cfg.get('schedule_tau', 1.0),
            scale=cfg.get('schedule_scale', 1.0),
        )

        self.register_buffer('action_mean', torch.tensor(self._action_mean))
        self.register_buffer('action_std', torch.tensor(self._action_std))

    ################### Training Setup ###################
    def configure_optimizers(self):
        '''
        配置优化器和学习率调度器：实现 SGDR + Peak Decay + Soft Restart
        '''
        # 1. 冻结不需要训练的模块 (保持原逻辑不变)
        if not self._train_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
        if not self._train_denoiser:
            for param in self.denoiser.parameters():
                param.requires_grad = False
        if self._with_predictor and (not self._train_predictor):
            for param in self.predictor.parameters():
                param.requires_grad = False

        params_to_update = []
        for param in self.parameters():
            if param.requires_grad: # 确保只添加需要梯度的参数
                params_to_update.append(param)

        assert len(params_to_update) > 0, 'No parameters to update'

        optimizer = bnb_optim.AdamW8bit(
            params_to_update,
            lr=self.cfg['lr'],
            weight_decay=self.cfg['weight_decay']
        )

        # 2. 获取基础调度参数
        accumulate = self.accumulate_grad_batches
        warmup_steps = self.cfg['lr_warmup_step'] // accumulate
        # total_steps 虽然在 lambda 中没直接用到，但用于计算 epoch 进度等
        # total_steps = self.cfg['lr_total_steps'] // accumulate 
        
        # 3. 获取 SGDR 参数 (建议在 yaml 中配置，这里提供了默认值)
        # 注意：cycle_len 也需要除以 accumulate，因为它是基于 step 计数的
        cycle_len = self.cfg.get('sgdr_cycle_len', 800) // accumulate
        cycle_mult = self.cfg.get('sgdr_cycle_mult', 1.5)
        decay_rate = self.cfg.get('sgdr_decay_rate', 0.85)
        min_lr_scale = self.cfg.get('sgdr_min_lr_scale', 0.01)
        cycle_warmup_ratio = self.cfg.get('sgdr_cycle_warmup_ratio', 0.1)

        def lr_lambda(current_step: int):
            # A. 全局预热 (Global Warmup)
            # 训练最开始的线性预热
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            
            # B. SGDR 阶段计算
            step_after_warmup = current_step - warmup_steps
            
            current_cycle_len = cycle_len
            cycle_idx = 0
            time_in_cycle = step_after_warmup
            
            # 迭代计算当前处于第几个周期，以及周期内的相对位置
            while time_in_cycle >= current_cycle_len:
                time_in_cycle -= current_cycle_len
                current_cycle_len = int(current_cycle_len * cycle_mult)
                cycle_idx += 1
            
            # C. 峰值衰减 (Peak Decay)
            # 随着周期数增加，最高 LR 逐渐降低
            peak_scale = decay_rate ** cycle_idx
            
            # D. 周期内调度逻辑
            
            # Case 1: 第 0 个周期 (Cycle 0)
            # 紧接在 Global Warmup 之后 (此时 LR=1.0)，因此不需要 Soft Restart
            # 直接从最高点开始进行余弦衰减，实现平滑过渡
            if cycle_idx == 0:
                progress = float(time_in_cycle) / float(max(1, current_cycle_len))
                cosine_value = 0.5 * (1.0 + math.cos(math.pi * progress))
                final_scale = peak_scale * cosine_value
            
            # Case 2: 后续周期 (Cycle > 0)
            # 经过了上一个周期的低谷，需要 Soft Restart (周期内热身) 重新拉高 LR
            else:
                cycle_warmup_steps = int(current_cycle_len * cycle_warmup_ratio)
                
                if time_in_cycle < cycle_warmup_steps:
                    # 周期内热身阶段: 0 -> Peak
                    restart_progress = float(time_in_cycle) / float(max(1, cycle_warmup_steps))
                    final_scale = peak_scale * restart_progress
                else:
                    # 周期内衰减阶段: Peak -> 0
                    # 有效衰减时间 = 总长 - 热身长
                    decay_steps = current_cycle_len - cycle_warmup_steps
                    time_in_decay = time_in_cycle - cycle_warmup_steps
                    
                    progress = float(time_in_decay) / float(max(1, decay_steps))
                    cosine_value = 0.5 * (1.0 + math.cos(math.pi * progress))
                    final_scale = peak_scale * cosine_value
            
            # E. 最低 LR 保护
            return max(final_scale, min_lr_scale)

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
    
    def process_anchors(self):
        """
        Process anchors for the model.
        """
        print(f"Loading anchors from {self.anchor_path}...")
        anchors_traj = np.load(self.anchor_path)  # [N, 40, 3]
        zeros_to_prepend = np.zeros((anchors_traj.shape[0], 1, anchors_traj.shape[-1]), dtype=anchors_traj.dtype)
        anchors_traj = np.concatenate([zeros_to_prepend, anchors_traj], axis=1)
        anchors_xy = anchors_traj[..., :2]  # [N, 41, 2]
        anchors_fine_grained_diffs = np.diff(anchors_xy, axis=1).astype(np.float32)  # [N, 40, 2]

        # 计算全局平均差分 (Global Mean) [B=1, 40, 2]
        global_mean_diff = np.mean(anchors_fine_grained_diffs, axis=0)
        self.register_buffer('global_mean_diff', torch.from_numpy(global_mean_diff).unsqueeze(0))

        self.num_anchors, self.num_diff_steps, self.diff_dim = anchors_fine_grained_diffs.shape

        anchor_diffs_flat = anchors_fine_grained_diffs.reshape(
            self.num_anchors, -1
        ).astype(np.float32)
        D_dim = anchor_diffs_flat.shape[1] # 80

        print(f"Building Faiss index (N={self.num_anchors}, D={D_dim})...")
        self.faiss_index = faiss.IndexFlatL2(D_dim) #
        self.faiss_index.add(anchor_diffs_flat) #
        print("Faiss index built.")

        print(f"Loading cluster data from {self.cluster_path}...")
        with open(self.cluster_path, 'rb') as f:
            cluster_data = pickle.load(f) #
            
        cluster_labels = cluster_data['labels'].astype(np.int64)

        cluster_center_diffs = cluster_data['cluster_centers_diffs'].astype(np.float32) # [K, 40, 2]
        zeros_to_prepend = np.zeros((cluster_center_diffs.shape[0], 1, cluster_center_diffs.shape[-1]), dtype=cluster_center_diffs.dtype)
        cluster_center_diffs = np.concatenate([zeros_to_prepend, cluster_center_diffs], axis=1)

        self.num_clusters = cluster_data['num_clusters']
        print(f"Loaded {self.num_clusters} clusters.")

        self.register_buffer('anchor_diffs', torch.from_numpy(anchors_fine_grained_diffs).cuda())
        self.register_buffer('cluster_diffs', torch.from_numpy(cluster_center_diffs).cuda())
        self.register_buffer('anchor_diffs_flat', torch.from_numpy(anchor_diffs_flat).cuda())
        self.register_buffer('cluster_labels', torch.from_numpy(cluster_labels).cuda())

        # ------------------- 新增：计算双重尺度统计量 -------------------
        
        # A. 计算 "无条件" 标准差 (std_uncond)
        #    即: 所有锚点相对于全局平均值的标准差
        #    Shape: [1, 40, 2]
        std_uncond = np.std(anchors_fine_grained_diffs - global_mean_diff, axis=0)
        # 防止除以0
        std_uncond[std_uncond < 1e-4] = 1.0
        
        # B. 计算 "有条件" 标准差 (std_cond)
        #    即: 每个锚点相对于它所属簇中心的标准差的平均值
        #    这是一个近似值，但足够有效。
        #    方法: 遍历所有簇，计算簇内方差，然后求平均。
        
        # 为了高效，我们可以直接计算 (Anchor - Cluster_Center) 的全局标准差
        # 获取每个锚点对应的簇中心
        anchor_cluster_centers = cluster_center_diffs[cluster_labels] # [N, 40, 2]
        anchor_residuals = anchors_fine_grained_diffs - anchor_cluster_centers
        
        std_cond = np.std(anchor_residuals, axis=0) # [40, 2]
        std_cond[std_cond < 1e-4] = 1.0

        print(f"Scale Analysis - Uncond Std Mean: {std_uncond.mean():.4f}, Cond Std Mean: {std_cond.mean():.4f}")

        self.register_buffer('std_uncond', torch.from_numpy(std_uncond).unsqueeze(0))
        self.register_buffer('std_cond', torch.from_numpy(std_cond).unsqueeze(0))
        # ----------------------------------------------------------------


    def forward(self, inputs, noised_actions_normalized, diffusion_step):
        """
        Forward pass of the VBD model.

        Args:
            inputs: Input data.
            noised_actions: noised actions.
            diffusion_step: Diffusion step.

        Returns:
            output_dict: Dictionary containing the model outputs.
        """
        inputs['anchors'] = self.anchor_tensor

        # Encode scene
        output_dict = {}
        encoder_outputs = self.encoder(inputs)

        if self._train_denoiser:
            denoiser_outputs = self.forward_denoiser(encoder_outputs, noised_actions_normalized, diffusion_step)
            output_dict.update(denoiser_outputs)

        if self._train_predictor:
            predictor_outputs = self.forward_predictor(encoder_outputs)
            output_dict.update(predictor_outputs)

        return output_dict

    def forward_denoiser(self, encoder_outputs, noised_inputs, diffusion_step, anchor_diff):
        """
        Forward pass of the denoiser module.

        Args:
            encoder_outputs: Outputs from the encoder module.
            noised_actions: noised actions.
            diffusion_step: Diffusion step.
            anchor_diff: Best Anchor TD.

        Returns:
            denoiser_outputs: Dictionary containing the denoiser outputs.
        """

        denoiser_output = self.denoiser(encoder_outputs, noised_inputs, diffusion_step, rollout=False)
        denoised_offset_norm = self.noise_scheduler.q_x0(
            denoiser_output,
            diffusion_step,
            noised_inputs,
            prediction_type=self._prediction_type
        )
        T_history_and_cur = encoder_outputs['T0']
        current_states = encoder_outputs['agents'][:, :self._agents_len, T_history_and_cur - 1]
        assert encoder_outputs['agents'].shape[1] >= self._agents_len, 'Too many agents to consider'

        # denoised_offset = self.unnormalize_anchor_increments(denoised_offset_norm)
        denoised_offset = denoised_offset_norm * self._offset_std

        # Roll out
        # When using decoder to predict the offset, the denoised_trajs is the original anchors + offset

        final_diff = denoised_offset + anchor_diff
        denoised_local_trajs, denoised_global_trajs = roll_out_new(
            current_states, final_diff, action_len=self._action_len
        )

        return {
            'denoiser_output': denoiser_output,
            'denoised_offset': denoised_offset,
            'denoised_local_trajs': denoised_local_trajs,
            'denoised_global_trajs': denoised_global_trajs,
        }

    def forward_conditioner(self, encoder_outputs, intent_command, target_cluster_center_diffs):
        """
        Forward pass of the conditioner module.

        Args:
            encoder_outputs: Outputs from the encoder module.

        Returns:
            predictor_outputs: Dictionary containing the predictor outputs.
        """
        # Normalize target cluster center diffs
        target_cluster_center_diffs_norm = (target_cluster_center_diffs - self.global_mean_diff) / self.std_uncond

        # Predict offset
        pred_norm_offsets, cluster_scores = self.predictor(encoder_outputs, intent_command, target_cluster_center_diffs_norm)

        # 构造掩码 [B, A, 1, 1]
        is_conditioned = (intent_command != 0).float().unsqueeze(-1).unsqueeze(-1)

        # 选择对应的标准差
        # 有条件 -> std_cond; 无条件 -> std_uncond
        # [B, A, 40, 2]
        # selected_std = (self.std_cond * is_conditioned) + \
        #                (self.std_uncond * (1.0 - is_conditioned))
        selected_std = self.std_uncond
        
        # 反归一化：将网络输出还原为物理尺度
        pred_offsets = pred_norm_offsets * selected_std

        # 有条件 -> 簇中心; 无条件 -> 全局平均
        batch_mean_prior = self.global_mean_diff.expand_as(target_cluster_center_diffs)
        
        selected_prior = (target_cluster_center_diffs * is_conditioned) + \
                         (batch_mean_prior * (1.0 - is_conditioned))
        
        # 重建局部绝对差分
        coarse_diffs = pred_offsets + selected_prior

        # current_states = encoder_outputs['agents'][:, :self._agents_len, -1]
        T_history_and_cur = encoder_outputs['T0']
        current_states = encoder_outputs['agents'][:, :self._agents_len, T_history_and_cur - 1]
        assert encoder_outputs['agents'].shape[1] >= self._agents_len, 'Too many agents to consider'

        # Roll out
        coarse_local_trajs, coarse_global_trajs = roll_out_new(
            current_states, coarse_diffs, action_len=self._action_len
        )

        return {
            'pred_offsets': pred_offsets,
            'pred_norm_offsets': pred_norm_offsets,
            'coarse_diffs': coarse_diffs,
            'coarse_local_trajs': coarse_local_trajs,
            'coarse_global_trajs': coarse_global_trajs,
            'cluster_scores': cluster_scores,
            'selected_std': selected_std,
            'selected_prior': selected_prior,
        }

    def forward_and_get_loss(self, batch, prefix='', debug=False):
        """
        Forward pass of the model and compute the loss.

        Args:
            batch: Input batch.
            prefix: Prefix for the loss keys.
            debug: Flag to enable debug mode.

        Returns:
            total_loss: Total loss.
            log_dict: Dictionary containing the loss values.
            debug_outputs: Dictionary containing debug outputs.
        """
        # data inputs
        agents_future = batch['agents_future']
        agents_future = agents_future[:, :, :self._future_len + 1, :]
        batch['agents_future'] = agents_future
        B, A_all, T_future_and_cur, D_all = agents_future.shape
        T_future_steps = T_future_and_cur // self._action_len
        D_predict = 2

        # TODO: Investigate why this to NAN
        # agents_future_valid = batch['agents_future_valid'][:, :self._agents_len]
        agents_future_valid = torch.ne(agents_future.sum(-1), 0)
        agents_future_valid = agents_future_valid[:, :, 1].unsqueeze(-1).expand_as(
            agents_future_valid) & agents_future_valid # B, A_all, T_future_and_cur
        agents_interested = batch['agents_interested']
        
        # --- 数据拼接：将历史和未来轨迹拼接为 agent 特征 ---
        agents_history = batch['agents_history']
        agents_features = torch.cat((agents_history[:, :, :-1, :5], agents_future[..., :5]), dim=-2)
        batch['agents_features'] = agents_features # B, A_all, T_history_and_cur + T_future_and_cur - 1, 5
        T_history_and_cur = agents_history.shape[-2]
        batch['T_history_and_cur'] = T_history_and_cur
        batch['T_history_and_cur'] = agents_history.shape[-2]

        agents_local = batch_transform_trajs_to_local_frame(agents_features, ref_idx=T_history_and_cur - 1)
        batch['agents_local'] = agents_local

        log_dict = {}
        debug_outputs = {}
        total_loss = 0


        ############## Run Encoder ##############
        encoder_outputs = self.encoder(batch)
        agents_future = agents_future[:, :self._agents_len] # B, A_pred, T_future_and_cur, 5
        agents_future_valid = agents_future_valid[:, :self._agents_len] # B, A_pred, T_future_and_cur
        agents_interested = agents_interested[:, :self._agents_len] # B, A_pred


        ############# Offset Calculation #############
        # --- Offset 计算：计算未来轨迹相对于锚点轨迹的偏移量 ---
        B_idx = torch.arange(B).unsqueeze(1)  # 生成形状为 [B, 1] 的批次索引
        A_idx = torch.arange(self._agents_len).unsqueeze(0)  # 生成形状为 [1, A_pred] 的车辆索引

        # B, A_pred, T_future_and_cur, 2
        gt_future_local = agents_local[:, :self._agents_len, T_history_and_cur - 1:, :D_predict]

        valid_sparse = agents_future_valid[:, :, ::self._action_len]  # B, A_pred, T_future_steps
        diff_valid_mask = valid_sparse[:, :, :-1] & valid_sparse[:, :, 1:] # [B, A_pred, T_future_steps]
        diff_valid_mask = diff_valid_mask.unsqueeze(-1) # [B, A_pred, T_future_steps, 1] 以便广播

        # B, A_pred, T_future_steps, 2
        gt_future_local_diff = torch.diff(gt_future_local[:, :, ::self._action_len, :], dim=-2)

        gt_masked = gt_future_local_diff.unsqueeze(2) * diff_valid_mask.unsqueeze(2)
        anchor_masked = self.anchor_diffs.unsqueeze(0).unsqueeze(0) * diff_valid_mask.unsqueeze(2)
        masked_diff = gt_masked - anchor_masked  # [B, A_pred, N, T_future_steps, 2]
        distances_sq = torch.sum(masked_diff**2, dim=[-1, -2]) # [B, A, 16384]

        nearest_fine_anchor_indices = torch.argmin(distances_sq, dim=2) # [B, A]

        # gt_future_diff = gt_future_diff * diff_valid_mask.float()
        # gt_future_local_diff_flat = gt_future_local_diff.reshape(B * self._agents_len, -1)

        # dists = torch.cdist(gt_future_local_diff_flat, self.anchor_diffs_flat) # [B*A, 16384]
        # nearest_fine_anchor_indices = torch.argmin(dists, dim=1)

        best_anchor_diff = self.anchor_diffs[nearest_fine_anchor_indices]  # B, A_pred, T_future_steps, 2
        target_cluster_indices = self.cluster_labels[nearest_fine_anchor_indices]
        target_cluster_center_diff = self.cluster_diffs[target_cluster_indices]  # B, A_pred, T_future_steps, 2

        target_cluster_to_gt_local_diff_offset = gt_future_local_diff - target_cluster_center_diff.view(B, self._agents_len, -1, self.diff_dim)  # B, A_pred, T_future_steps, 2
        target_cluster_to_gt_local_diff_offset = target_cluster_to_gt_local_diff_offset * diff_valid_mask.float()


        ############### Behavior Prior Prediction #################
        if self._train_predictor:
            mask_shape = (B, self._agents_len)
            dropout_mask = (torch.rand(mask_shape, device=self.device) < self._intent_dropout)
            intent_command = torch.where(dropout_mask, 0, target_cluster_indices.view(B, self._agents_len))

            goal_outputs = self.forward_conditioner(encoder_outputs, intent_command, target_cluster_center_diff.view(B, self._agents_len, -1, self.diff_dim))
            debug_outputs.update(goal_outputs)

            # get loss
            coarse_offsets = goal_outputs['pred_offsets'] # B, A_pred, T_future, 5
            coarse_norm_offsets = goal_outputs['pred_norm_offsets'] # B, A_pred, T_future, 5
            coarse_diffs = goal_outputs['coarse_diffs']
            coarse_local_trajs = goal_outputs['coarse_local_trajs'] # B, A_pred, T_future, 5
            cluster_scores = goal_outputs['cluster_scores'] # B, A_pred, num_clusters
            selected_std = goal_outputs['selected_std']
            selected_prior = goal_outputs['selected_prior']

            target_offset = (gt_future_local_diff - selected_prior) * diff_valid_mask.float()
            target_norm_offset = target_offset / selected_std
            target_norm_offset = torch.clamp(target_norm_offset, min=-3.0, max=3.0)

            coarse_traj_loss, cluster_score_loss = self.goal_loss(coarse_norm_offsets, cluster_scores, 
                                              target_norm_offset, target_cluster_indices,
                                              diff_valid_mask, agents_interested)

            pred_loss = self.goal_loss_weight * coarse_traj_loss + self.score_loss_weight * cluster_score_loss
            total_loss += self.predictor_loss_weight * pred_loss

            pred_ade, pred_fde = self.calculate_metrics_predict(
                coarse_local_trajs, gt_future_local, agents_future_valid, agents_interested, 8
            )
            # pred_ade, pred_fde = self.calculate_metrics_predict_new(
            #     goal_trajs, agents_future, agents_future_valid, agents_interested, 16
            # )

            log_dict.update({
                prefix + 'goal_loss': coarse_traj_loss.item(),
                prefix + 'score_loss': cluster_score_loss.item(),
                prefix + 'pred_ADE': pred_ade,
                prefix + 'pred_FDE': pred_fde,
            })

        ################### 采样 (Scheduled Sampling) ###################
        pred_coarse_diff = coarse_diffs.view(B, self._agents_len, -1, self.diff_dim)

        if self._balanced_input == 'prob':
            # 2. 生成 Per-Agent 随机掩码 [B, A_pred]
            #    True = 使用真值锚点, False = 使用预测轨迹
            use_anchor_mask = torch.rand(B, self._agents_len, device=self.device) < self._input_weight
            
            #    调整形状以广播: [B, A, 1, 1]
            use_anchor_mask = use_anchor_mask.view(B, self._agents_len, 1, 1)

            # 3. 准备候选项
            #    Candidate A: 真值最佳锚点 (训练初期目标)
            candidate_anchor = best_anchor_diff.view(B, self._agents_len, -1, self.diff_dim)
            
            #    Candidate B: Predictor 预测出的粗略轨迹
            #    [重要] 必须 .detach()，防止 Denoiser "教" Predictor 作弊
            candidate_pred = coarse_diffs.view(B, self._agents_len, -1, self.diff_dim).detach()

            # 4. 执行采样 (物理上真实存在的轨迹，而不是平均值)
            #    balanced_diff_input 现在是 "Base" (基准轨迹)
            balanced_diff_input = torch.where(use_anchor_mask, candidate_anchor, candidate_pred)

        elif self._balanced_input == 'linear':
            balanced_diff_input = best_anchor_diff.view(B, self._agents_len, -1, self.diff_dim) * self._input_weight + pred_coarse_diff * (1 - self._input_weight)

        target_anchor_to_gt_local_diff_offset = gt_future_local_diff - balanced_diff_input  # B, A_pred, T_future_steps, 2
        target_anchor_to_gt_local_diff_offset = target_anchor_to_gt_local_diff_offset * diff_valid_mask.float()

        ############### Denoise #################
        if self._train_denoiser:
            # get predicted anchor
            assert cluster_scores != None, 'No valid goal predictions yet.'

            diffusion_steps = torch.randint(
                1, self.noise_scheduler.num_steps * 1 // 20, (B, self._agents_len),
                device=agents_future.device
            ).long()
            diffusion_steps = diffusion_steps.view(B, self._agents_len, 1, 1)

            noise = torch.randn_like(target_anchor_to_gt_local_diff_offset) # B, A_pred, T_future_steps, 2

            # target_offset_norm = self.normalize_anchor_increments(target_anchor_to_gt_local_diff_offset)
            target_offset_norm = target_anchor_to_gt_local_diff_offset / self._offset_std

            noised_target_offset_norm = self.noise_scheduler.add_noise(
                target_offset_norm,
                noise,
                diffusion_steps
            )
            # noised_target_offset_norm = torch.clamp(noised_target_offset_norm, min=-1, max=1)

            # noised_target_offset = self.unnormalize_anchor_increments(noised_target_offset_norm) # B, A_pred, T_future_steps, 2

            # Inverse diffusion
            denoise_outputs = self.forward_denoiser(encoder_outputs, noised_target_offset_norm,
                                                    diffusion_steps.view(B, self._agents_len), 
                                                    balanced_diff_input.view(B, self._agents_len, -1, self.diff_dim))
            # denoise_outputs['denoiser_output']: B, A_pred, T_future_steps, 2
            # denoise_outputs['denoised_offset']: B, A_pred, T_future_steps, 2
            # denoise_outputs['denoised_local_trajs']: B, A_pred, T_future, 5
            # denoise_outputs['denoised_trajs_origin']: B, A_pred, T_future, 5

            debug_outputs.update(denoise_outputs)
            debug_outputs['noise'] = noise
            debug_outputs['diffusion_steps'] = diffusion_steps

            # Get Loss
            denoised_local_trajs = denoise_outputs['denoised_local_trajs']
            denoised_offset = denoise_outputs['denoised_offset']
            if self._prediction_type == 'sample':
                diff_loss = self.diff_loss(
                    denoised_offset, target_anchor_to_gt_local_diff_offset, diff_valid_mask, agents_interested
                )
                total_loss += self.denoiser_loss_weight * (self.traj_loss_weight * diff_loss)

                # Predict the noise
                # _, diffusion_loss = self.noise_scheduler.get_noise(
                #     x_0=denoise_outputs['denoised_actions_normalized'],
                #     x_t=noised_action_normalized,
                #     timesteps=diffusion_steps,
                #     gt_noise=noise,
                # )

                log_dict.update({
                    prefix + 'traj_loss': diff_loss.item(),
                    # prefix + 'diffusion_loss': diffusion_loss.item()
                })

            elif self._prediction_type == 'error':
                denoiser_output = denoise_outputs['denoiser_output']
                # denoise_loss = torch.nn.functional.mse_loss(
                #     denoiser_output, noise, reduction='mean'
                # )
                denoise_loss = smooth_l1_loss(denoiser_output, noise, reduction='none')

                agent_mask = agents_interested.unsqueeze(-1).unsqueeze(-1) > 0
                denoise_loss = denoise_loss * agent_mask
                denoise_loss = denoise_loss.sum() / (agent_mask.sum() * T_future_steps * D_predict + 1e-6)
                total_loss += self.denoiser_loss_weight * (self.denoise_loss_weight * denoise_loss)
                log_dict.update({
                    prefix + 'diffusion_loss': denoise_loss.item(),
                })

            elif self._prediction_type == 'mean':
                # pred_action_normalized = denoise_outputs['denoised_actions_normalized']
                # denoise_loss = self.action_loss(
                #     pred_action_normalized, gt_actions_normalized, gt_actions_valid, agents_interested
                # )
                denoise_loss = self.traj_loss(
                    denoised_local_trajs, agents_future, agents_future_valid, agents_interested
                )
                total_loss += (self.denoiser_loss_weight * self.denoise_loss_weight * denoise_loss)
                # log_dict.update({
                #     prefix + 'action_loss': denoise_loss.item(),
                # })
                log_dict.update({
                    prefix + 'state_loss': denoise_loss.item(),
                })
            else:
                raise ValueError('Invalid prediction type')

            denoise_ade, denoise_fde = self.calculate_metrics_denoise(
                denoised_local_trajs, gt_future_local, agents_future_valid, agents_interested, 8
            )

            log_dict.update({
                prefix + 'denoise_ADE': denoise_ade,
                prefix + 'denoise_FDE': denoise_fde,
            })

        log_dict['input_weight'] = self._input_weight
        log_dict['rank_weight'] = self.rank_loss_weight
        log_dict[prefix + 'loss'] = total_loss.item()

        if debug:
            return total_loss, log_dict, debug_outputs
        else:
            return total_loss, log_dict

    def training_step(self, batch, batch_idx):
        """
        Training step of the model.

        Args:
            batch: Input batch.
            batch_idx: Batch index.

        Returns:
            loss: Loss value.
        """
        # batch['agents_history']: B, A_all, T_history_and_cur, 8
        # batch['agents_interested']: B, A_all
        # batch['agents_future']: B, A_all, T_future_and_cur, 5
        # batch['agents_type']: B, A_all
        # batch['traffic_light_points']: B, N_tls, 3
        # batch['polylines']: B, N_polylines, T_polyline, 5
        # batch['polylines_valid']: B, N_polylines
        # batch['relations']: B, A_all + N_tls + N_polylines, A_all + N_tls + N_polylines, 3

        global_step = self.global_step

        if self._balanced_input != 'no':
            if global_step >= self._weight_anneal_steps // self.accumulate_grad_batches:
                self._input_weight = self._weight_end
            else:
                weight_decay = (self._weight_start - self._weight_end) * (
                    1 - global_step / (self._weight_anneal_steps // self.accumulate_grad_batches)
                )
                self._input_weight = self._weight_end + weight_decay
        else:
            self._input_weight = self._weight_end

        if self.use_dynamic_rank_weight:
            if global_step < self.rank_weight_anneal_steps // self.accumulate_grad_batches:
                progress = global_step / (self.rank_weight_anneal_steps // self.accumulate_grad_batches)
                self.rank_loss_weight = self.rank_weight_start + (self.rank_weight_end - self.rank_weight_start) * progress
            else:
                self.rank_loss_weight = self.rank_weight_end
        else:
            self.rank_loss_weight = self.rank_weight_end

        loss, log_dict = self.forward_and_get_loss(batch, prefix='train/')
        self.log_dict(
            log_dict,
            on_step=True, on_epoch=False, sync_dist=True,
            prog_bar=True
        )

        return loss

    def validation_step(self, batch, batch_idx):
        """
        Validation step of the model.

        Args:
            batch: Input batch.
            batch_idx: Batch index.
        """
        loss, log_dict = self.forward_and_get_loss(batch, prefix='val/')
        self.log_dict(log_dict,
                      on_step=False, on_epoch=True, sync_dist=True,
                      prog_bar=True)

        return loss

    ################### Loss function ###################
    def denoise_loss(
            self, denoised_trajs,
            agents_future, agents_future_valid,
            agents_interested
    ):
        """
        Calculates the denoise loss for the denoised actions and trajectories.

        Args:
            denoised_actions_normalized (torch.Tensor): Normalized denoised actions tensor of shape [B, A, T, C].
            denoised_trajs (torch.Tensor): Denoised trajectories tensor of shape [B, A, T, C].
            agents_future (torch.Tensor): Future agent positions tensor of shape [B, A, T, 3].
            agents_future_valid (torch.Tensor): Future agent validity tensor of shape [B, A, T].
            gt_actions_normalized (torch.Tensor): Normalized ground truth actions tensor of shape [B, A, T, C].
            gt_actions_valid (torch.Tensor): Ground truth actions validity tensor of shape [B, A, T].
            agents_interested (torch.Tensor): Interested agents tensor of shape [B, A].

        Returns:
            state_loss_mean (torch.Tensor): Mean state loss.
            yaw_loss_mean (torch.Tensor): Mean yaw loss.
            action_loss_mean (torch.Tensor): Mean action loss.
        """

        agents_future = agents_future[..., 1:, :3]
        future_mask = agents_future_valid[..., 1:] * (agents_interested[..., None] > 0)

        # Calculate State Loss
        # [B, A, T]
        state_loss = smooth_l1_loss(denoised_trajs[..., :2], agents_future[..., :2], reduction='none').sum(-1)
        yaw_error = (denoised_trajs[..., 2] - agents_future[..., 2])
        yaw_error = torch.atan2(torch.sin(yaw_error), torch.cos(yaw_error))
        yaw_loss = torch.abs(yaw_error)

        # Filter out the invalid state
        state_loss = state_loss * future_mask
        yaw_loss = yaw_loss * future_mask

        # Calculate the mean loss
        state_loss_mean = state_loss.sum() / future_mask.sum()
        yaw_loss_mean = yaw_loss.sum() / future_mask.sum()

        return state_loss_mean, yaw_loss_mean

    def action_loss(
            self, actions, actions_gt, actions_valid, agents_interested
    ):
        """
        Calculates the loss for action prediction.

        Args:
            actions (torch.Tensor): Tensor of shape [B, A, T, 2] representing predicted actions.
            actions_gt (torch.Tensor): Tensor of shape [B, A, T, 2] representing ground truth actions.
            actions_valid (torch.Tensor): Tensor of shape [B, A, T] representing validity of actions.
            agents_interested (torch.Tensor): Tensor of shape [B, A] representing interest in agents.

        Returns:
            action_loss_mean (torch.Tensor): Mean action loss.
        """
        # Get Mask
        action_mask = actions_valid * (agents_interested[..., None] > 0)

        # Calculate the action loss
        action_loss = smooth_l1_loss(actions, actions_gt, reduction='none').sum(-1)
        action_loss = action_loss * action_mask

        # Calculate the mean loss
        action_loss_mean = action_loss.sum() / action_mask.sum()

        return action_loss_mean

    def traj_loss(
            self, trajs, trajs_gt, trajs_valid, agents_interested
    ):
        """
        Calculates the loss for trajectory prediction.

        Args:
            trajs (torch.Tensor): Tensor of shape [B, A_pred, T_future, 5] representing predicted trajectories.
            trajs_gt (torch.Tensor): Tensor of shape [B, A_pred, T_future_and_cur, 5] representing ground truth trajectories.
            trajs_valid (torch.Tensor): Tensor of shape [B, A_pred, T_future_and_cur] representing validity of trajectories.
            agents_interested (torch.Tensor): Tensor of shape [B, A_pred] representing interest in agents.

        Returns:
            trajs_loss_mean (torch.Tensor): Mean trajectory loss.
        """
        # Get Mask
        trajs_mask = trajs_valid[:, :, 1:] * (agents_interested[..., None] > 0)

        # Calculate the trajs loss
        trajs_loss = smooth_l1_loss(trajs[..., :2], trajs_gt[:, :, 1:, :2], reduction='none').sum(-1)
        trajs_loss = trajs_loss * trajs_mask

        # Calculate the mean loss
        trajs_loss_mean = trajs_loss.sum() / trajs_mask.sum()

        return trajs_loss_mean
    
    def diff_loss(
            self, diffs, diffs_gt, diffs_valid, agents_interested
    ):
        """
        Calculates the loss for trajectory difference prediction.

        Args:
            diffs (torch.Tensor): Tensor of shape [B, A_pred, T_future_steps, 2] representing predicted trajectory differences.
            diffs_gt (torch.Tensor): Tensor of shape [B, A_pred, T_future_steps, 2] representing ground truth trajectory differences.
            diffs_valid (torch.Tensor): Tensor of shape [B, A_pred, T_future_steps] representing validity of trajectory differences.
            agents_interested (torch.Tensor): Tensor of shape [B, A_pred] representing interest in agents.

        Returns:
            diff_loss_mean (torch.Tensor): Mean difference loss.
        """
        # Get Mask
        diff_mask = diffs_valid.squeeze(-1) * (agents_interested.unsqueeze(-1) > 0)

        # Calculate the diff loss
        diff_loss = smooth_l1_loss(diffs, diffs_gt, reduction='none').sum(-1)
        diff_loss = diff_loss * diff_mask

        # Calculate the mean loss
        diff_loss_mean = diff_loss.sum() / diff_mask.sum()

        return diff_loss_mean

    def goal_loss(
            self, coarse_diffs, cluster_scores, gt_diffs, target_indices,
            diff_valid_mask, agents_interested
    ):
        """
        Calculates the loss for trajectory prediction (IntentConditioner).
        
        Updated for CFG: Now calculates regression loss on 'coarse_diffs' (absolute) 
        against 'gt_diffs' (absolute), ensuring consistent objectives for both 
        conditional and unconditional passes.

        Args:
            coarse_diffs (torch.Tensor): Predicted absolute trajectory differences, shape [B, A_pred, T_future, 2].
            cluster_scores (torch.Tensor): Predicted cluster scores (logits), shape [B, A_pred, num_clusters].
            gt_diffs (torch.Tensor): Ground truth absolute trajectory differences, shape [B, A_pred, T_future, 2].
            target_indices (torch.Tensor): Ground truth cluster indices, shape [B, A_pred].
            diff_valid_mask (torch.Tensor): Validity mask for diffs, shape [B, A_pred, T_future, 1].
            agents_interested (torch.Tensor): Interest mask, shape [B, A_pred].

        Returns:
            traj_loss_mean (torch.Tensor): Mean regression loss.
            score_loss_mean (torch.Tensor): Mean score loss.
        """
        agents_interested_flat = agents_interested.flatten().float()  # [B * A_pred]

        # --- 1. Calculate Score Loss (Classification) ---
        score_loss = 0.0
        num_query = cluster_scores.shape[-1]

        # BCE Loss Calculation (Supports 'bce' or 'mix')
        if self.score_loss_type in ['bce', 'mix']:
            bce_target = torch.nn.functional.one_hot(target_indices, num_classes=num_query).float()
            # Multi-label BCE (though here it's single label target)
            bce_loss = binary_cross_entropy_with_logits(cluster_scores, bce_target, reduction='none').sum(dim=-1)
            score_loss += self.bce_loss_weight * bce_loss.view(-1)  # [B * A_pred]

        # (Rank/Hinge loss logic removed/commented out as standard CrossEntropy/BCE is usually sufficient for intent classification)
        
        # Average the score loss over valid agents
        score_loss = score_loss * (agents_interested_flat > 0).float()
        score_loss_mean = score_loss.sum() / torch.clamp((agents_interested_flat > 0).sum(), min=1.0)

        # --- 2. Calculate Regression Loss (Absolute Diffs) ---
        # Direct L1 loss between predicted absolute diffs and GT absolute diffs.
        # This works for both:
        #   - Conditional case: (Pred_Offset + Cluster) approx GT
        #   - Unconditional case: (Pred_Offset + 0) approx GT
        
        traj_loss = smooth_l1_loss(coarse_diffs, gt_diffs, reduction='none').sum(-1) # [B, A, T]
        
        # Combine masks: Trajectory Validity AND Agent Interest
        # diff_valid_mask: [B, A, T, 1] -> squeeze -> [B, A, T]
        # agents_interested: [B, A] -> unsqueeze -> [B, A, 1]
        valid_mask = diff_valid_mask.squeeze(-1) * (agents_interested.unsqueeze(-1) > 0)
        
        traj_loss = traj_loss * valid_mask.float()
        traj_loss_mean = traj_loss.sum() / torch.clamp(valid_mask.sum(), min=1.0)

        # --- 3. Return losses ---
        return traj_loss_mean, score_loss_mean

    @torch.no_grad()
    def calculate_metrics_denoise(self,
                                  denoised_trajs, agents_future, agents_future_valid,
                                  agents_interested, top_k=None
                                  ):
        """
        Calculates the denoising metrics for the predicted trajectories.

        Args:
            denoised_trajs (torch.Tensor): Denoised trajectories of shape [B, A, T, 2].
            agents_future (torch.Tensor): Ground truth future trajectories of agents of shape [B, A, T, 2].
            agents_future_valid (torch.Tensor): Validity mask for future trajectories of agents of shape [B, A, T].
            agents_interested (torch.Tensor): Interest mask for agents of shape [B, A].
            top_k (int, optional): Number of top agents to consider. Defaults to None.

        Returns:
            Tuple[float, float]: A tuple containing the denoising ADE (Average Displacement Error) and FDE (Final Displacement Error).
        """

        if not top_k:
            top_k = self._agents_len

        pred_traj = denoised_trajs[:, :top_k, :, :2]  # [B, A, T, 2]
        gt = agents_future[:, :top_k, 1:, :2]  # [B, A, T, 2]
        gt_mask = (agents_future_valid[:, :top_k, 1:] \
                   & (agents_interested[:, :top_k, None] > 0)).bool()  # [B, A, T]

        denoise_mse = torch.norm(pred_traj - gt, dim=-1)
        denoise_ADE = denoise_mse[gt_mask].mean()
        denoise_FDE = denoise_mse[..., -1][gt_mask[..., -1]].mean()

        return denoise_ADE.item(), denoise_FDE.item()

    @torch.no_grad()
    def calculate_metrics_predict(self,
                                  goal_trajs, agents_future, agents_future_valid,
                                  agents_interested, top_k=None
                                  ):
        """
        Calculates the metrics for predicting goal trajectories.

        Args:
            goal_trajs (torch.Tensor): Tensor of shape [B, A, Q, T, 2] representing the goal trajectories.
            agents_future (torch.Tensor): Tensor of shape [B, A, T, 2] representing the future trajectories of agents.
            agents_future_valid (torch.Tensor): Tensor of shape [B, A, T] representing the validity of future trajectories.
            agents_interested (torch.Tensor): Tensor of shape [B, A] representing the interest level of agents.
            top_k (int, optional): The number of top agents to consider. Defaults to None.

        Returns:
            tuple: A tuple containing the goal Average Displacement Error (ADE) and goal Final Displacement Error (FDE).
        """

        if not top_k:
            top_k = self._agents_len
        goal_trajs = goal_trajs[:, :top_k, :, :2]  # [B, A, T, 2]
        gt = agents_future[:, :top_k, 1:, :2]  # [B, A, T, 2]
        gt_mask = (agents_future_valid[:, :top_k, 1:]
                   & (agents_interested[:, :top_k, None] > 0)).bool()  # [B, A, T]

        goal_mse = torch.norm(goal_trajs - gt[:, :, :, :], dim=-1)  # [B, A, T]
        goal_mse = goal_mse * gt_mask  # [B, A, T]

        goal_ADE = goal_mse.sum() / gt_mask.sum()
        goal_FDE = goal_mse[..., -1].sum() / gt_mask[..., -1].sum()

        return goal_ADE.item(), goal_FDE.item()

    ################### Helper Functions ##############
    def batch_to_device(self, input_dict: dict, device: torch.device = 'cuda'):
        """
        Move the tensors in the input dictionary to the specified device.

        Args:
            input_dict (dict): A dictionary containing tensors to be moved.
            device (torch.device): The target device to move the tensors to.

        Returns:
            dict: The input dictionary with tensors moved to the specified device.
        """
        for key, value in input_dict.items():
            if isinstance(value, torch.Tensor):
                input_dict[key] = value.to(device)

        return input_dict

    def normalize_actions(self, actions: torch.Tensor):
        """
        Normalize the given actions using the mean and standard deviation.

        Args:
            actions : The actions to be normalized.

        Returns:
            The normalized actions.
        """
        return (actions - self.action_mean) / self.action_std

    def unnormalize_actions(self, actions: torch.Tensor):
        """
        Unnormalize the given actions using the stored action standard deviation and mean.

        Args:
            actions: The normalized actions to be unnormalized.

        Returns:
             The unnormalized actions.
        """
        return actions * self.action_std + self.action_mean
    
    def normalize_anchor_increments(self, anchors: torch.Tensor):
        """
        Normalize the given anchor increments using the min and max values.

        Args:
            anchors : The anchor increments to be normalized. Shape [B, N, T, 2].
        Returns:
            The normalized anchor increments.
        """
        if not self._normalize_anchors:
            return anchors
        
        # self.anchor_incre_min is [x_min, y_min], self.anchor_incre_max is [x_max, y_max]
        min_val = torch.tensor(self.anchor_incre_min, device=anchors.device, dtype=anchors.dtype)
        max_val = torch.tensor(self.anchor_incre_max, device=anchors.device, dtype=anchors.dtype)
        
        # Min-max normalization to [-1, 1]
        # min_val and max_val are shape [2], they broadcast to [B, N, T, 2]
        return 2 * (anchors - min_val) / (max_val - min_val) - 1
        

    def unnormalize_anchor_increments(self, anchors: torch.Tensor):
        """
        Unnormalize the given anchor increments using the stored anchor increment min and max values.

        Args:
            anchors: The normalized anchor increments to be unnormalized.

        Returns:
             The unnormalized anchor increments.
        """
        if not self._normalize_anchors:
            return anchors
        
        min_val = torch.tensor(self.anchor_incre_min, device=anchors.device, dtype=anchors.dtype)
        max_val = torch.tensor(self.anchor_incre_max, device=anchors.device, dtype=anchors.dtype)
        
        # Inverse of min-max normalization from [-1, 1]
        return (anchors + 1) * (max_val - min_val) / 2 + min_val

