import numpy as np
import torch
import lightning.pytorch as pl
from .modules_v2 import Encoder, Denoiser, GoalPredictor
from .utils import DDPM_Sampler
from .model_utils_new import (inverse_kinematics, roll_out, batch_transform_trajs_to_global_frame,
                              get_trajectory_type, interpolate_anchors, roll_out_new)
from torch.nn.functional import smooth_l1_loss, cross_entropy, gumbel_softmax
import math


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
        self._task_probabilities = cfg.get('task_probabilities', None)
        self.anchor_incre_min = cfg['anchor_incre_min']
        self.anchor_incre_max = cfg['anchor_incre_max']
        self._gumbel_tau = cfg.get('gumbel_tau', 1.0)

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
        self.anchor = np.load('/home/karim/VBD_host/vbd/data/kmeans_navsim_traj_20.npy')
        self.anchor = interpolate_anchors(self.anchor, self._future_len + 1)
        self.anchor_tensor = torch.tensor(self.anchor, dtype=torch.float32).to('cuda')

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
            self.predictor = GoalPredictor(
                future_len=self._future_len,
                agents_len=self._agents_len,
                action_len=self._action_len,
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
        This function is called by Lightning to create the optimizer and learning rate scheduler.
        '''
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
            # if param.requires_grad == True:
            #     params_to_update.append(param)
            params_to_update.append(param)

        assert len(params_to_update) > 0, 'No parameters to update'

        optimizer = torch.optim.AdamW(
            params_to_update,
            lr=self.cfg['lr'],
            weight_decay=self.cfg['weight_decay']
        )

        warmup_steps = self.cfg['lr_warmup_step']
        total_steps = self.cfg['lr_total_steps']
        end_factor = self.cfg.get('lr_end_factor', 0.01)

        def lr_lambda(current_step: int):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            
            progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            
            # Cosine annealing schedule
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            
            # Mix with linear decay to the end_factor
            return (1.0 - end_factor) * cosine_decay + end_factor

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

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

    def forward_denoiser(self, encoder_outputs, noised_anchors_gt, diffusion_step):
        """
        Forward pass of the denoiser module.

        Args:
            encoder_outputs: Outputs from the encoder module.
            noised_actions: noised actions.
            diffusion_step: Diffusion step.

        Returns:
            denoiser_outputs: Dictionary containing the denoiser outputs.
        """
        # noised_actions = self.unnormalize_actions(noised_actions_normalized)
        # denoiser_output = self.denoiser(encoder_outputs, noised_actions, diffusion_step)
        denoiser_output = self.denoiser(encoder_outputs, noised_anchors_gt, diffusion_step, rollout=False)
        # denoised_traj_increments = self.noise_scheduler.q_x0(
        #     denoiser_output,
        #     diffusion_step,
        #     noised_anchors_gt,
        #     prediction_type=self._prediction_type
        # )
        # current_states = encoder_outputs['agents'][:, :self._agents_len, -1]
        T_history_and_cur = encoder_outputs['T0']
        current_states = encoder_outputs['agents'][:, :self._agents_len, T_history_and_cur - 1]
        assert encoder_outputs['agents'].shape[1] >= self._agents_len, 'Too many agents to consider'

        # Roll out
        # denoised_actions = self.unnormalize_actions(denoised_actions_normalized)
        # denoised_trajs = roll_out(current_states, denoised_actions,
        #                           action_len=self.denoiser._action_len, dt=self._dt, global_frame=True)

        # When using decoder to predict the offset, the denoised_trajs is the original anchors + offset
        denoised_trajs, denoised_trajs_origin = roll_out_new(
            current_states, denoiser_output, global_frame=True)


        # return {
        #     'denoiser_output': denoiser_output,
        #     'denoised_actions_normalized': denoised_actions_normalized,
        #     'denoised_actions': denoised_actions,
        #     'denoised_trajs': denoised_trajs,
        # }
        return {
            'denoiser_output': denoiser_output,
            'denoised_trajs': denoised_trajs,
            'denoised_trajs_origin': denoised_trajs_origin,
        }

    def forward_predictor(self, encoder_outputs):
        """
        Forward pass of the predictor module.

        Args:
            encoder_outputs: Outputs from the encoder module.

        Returns:
            predictor_outputs: Dictionary containing the predictor outputs.
        """
        # Predict goal
        goal_actions_normalized, goal_scores = self.predictor(encoder_outputs)
        # goal_actions_normalized, goal_types = self.predictor(encoder_outputs)

        # current_states = encoder_outputs['agents'][:, :self._agents_len, -1]
        T_history_and_cur = encoder_outputs['T0']
        current_states = encoder_outputs['agents'][:, :self._agents_len, T_history_and_cur - 1]
        assert encoder_outputs['agents'].shape[1] >= self._agents_len, 'Too many agents to consider'

        # Roll out
        goal_actions = self.unnormalize_actions(goal_actions_normalized)
        # goal_trajs = roll_out(current_states[:, :, None, :], goal_actions,
        #             action_len=self.predictor._action_len, global_frame=True)
        goal_trajs = roll_out(current_states[:, :, None, :], goal_actions,
                              action_len=self.predictor._action_len, global_frame=True)

        # return {
        #     'goal_actions_normalized': goal_actions_normalized,
        #     'goal_actions': goal_actions,
        #     'goal_scores': goal_scores,
        #     'goal_trajs': goal_trajs,
        # }
        return {
            'goal_actions_normalized': goal_actions_normalized,
            'goal_actions': goal_actions,
            'goal_scores': goal_scores,
            # 'goal_types': goal_types,
            'goal_trajs': goal_trajs,
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
        B, A_all, T_future, D_all = agents_future.shape
        T_future_steps = T_future // self._action_len
        D_predict = 2
        batch['anchors'] = self.anchor_tensor.unsqueeze(0).unsqueeze(0).expand(B, self._agents_len, -1, -1, -1)

        # TODO: Investigate why this to NAN
        # agents_future_valid = batch['agents_future_valid'][:, :self._agents_len]
        agents_future_valid = torch.ne(agents_future.sum(-1), 0)
        agents_future_valid = agents_future_valid[:, :, 1].unsqueeze(-1).expand_as(
            agents_future_valid) & agents_future_valid
        agents_interested = batch['agents_interested']
        anchors = batch['anchors']

        # # get actions from trajectory
        # gt_actions, gt_actions_valid = inverse_kinematics(
        #     agents_future,
        #     agents_future_valid,
        #     dt=self._dt,
        #     action_len=self._action_len,
        # )
        #
        # gt_actions_normalized = self.normalize_actions(gt_actions)
        # B, A, T, D = gt_actions_normalized.shape
        
        # --- 数据拼接：将历史和未来轨迹拼接为 agent 特征 ---
        agents_history = batch['agents_history']
        agents_features = torch.cat((agents_history[:, :, :-1, :5], agents_future[..., :5]), dim=-2)
        batch['agents_features'] = agents_features
        T_history_and_cur = agents_history.shape[-2]
        batch['T_history_and_cur'] = T_history_and_cur
        batch['T_history_and_cur'] = agents_history.shape[-2]

        log_dict = {}
        debug_outputs = {}
        total_loss = 0

        goal_scores = None

        ############## Run Encoder ##############
        encoder_outputs = self.encoder(batch)
        agents_future = agents_future[:, :self._agents_len]
        agents_future_valid = agents_future_valid[:, :self._agents_len]
        agents_interested = agents_interested[:, :self._agents_len]

        ############### Behavior Prior Prediction #################
        if self._train_predictor:
            goal_outputs = self.forward_predictor(encoder_outputs)
            debug_outputs.update(goal_outputs)

            # get loss
            goal_scores = goal_outputs['goal_scores']
            # goal_types = goal_outputs['goal_types']
            goal_trajs = goal_outputs['goal_trajs']

            goal_loss_mean, score_loss_mean = self.goal_loss(
                goal_trajs, goal_scores, agents_future,
                agents_future_valid, anchors,
                agents_interested,
            )
            # goal_loss_mean, type_loss_mean = self.goal_loss_new(
            #     goal_trajs, goal_types, agents_future,
            #     agents_future_valid, agents_interested
            # )

            pred_loss = self.goal_loss_weight * goal_loss_mean + self.score_loss_weight * score_loss_mean
            # pred_loss = goal_loss_mean + 0.05 * type_loss_mean
            total_loss += self.predictor_loss_weight * pred_loss

            pred_ade, pred_fde = self.calculate_metrics_predict(
                goal_trajs, agents_future, agents_future_valid, agents_interested, 8
            )
            # pred_ade, pred_fde = self.calculate_metrics_predict_new(
            #     goal_trajs, agents_future, agents_future_valid, agents_interested, 16
            # )

            log_dict.update({
                prefix + 'goal_loss': goal_loss_mean.item(),
                prefix + 'score_loss': score_loss_mean.item(),
                prefix + 'pred_ADE': pred_ade,
                prefix + 'pred_FDE': pred_fde,
            })
            # log_dict.update({
            #     prefix + 'goal_loss': goal_loss_mean.item(),
            #     prefix + 'score_loss': type_loss_mean.item(),
            #     prefix + 'pred_ADE': pred_ade,
            #     prefix + 'pred_FDE': pred_fde,
            # })


        ############### Denoise #################
        if self._train_denoiser:
            # get predicted anchor
            assert goal_scores != None, 'No valid goal predictions yet.'
            B_idx = torch.arange(B).unsqueeze(1)  # 生成形状为 [B, 1] 的批次索引
            A_idx = torch.arange(self._agents_len).unsqueeze(0)  # 生成形状为 [1, A] 的车辆索引
            # Use Gumbel-Softmax for differentiable sampling of anchors
            goal_one_hot = gumbel_softmax(goal_scores, tau=self._gumbel_tau, hard=True, dim=-1)

            # Introduce random exploration
            batch_index_mask = torch.rand(B, device=goal_scores.device) < self._random_target
            if batch_index_mask.any():
                num_anchors = goal_scores.shape[-1]
                random_indices = torch.randint(0, num_anchors, (B, self._agents_len), device=goal_scores.device)
                random_one_hot = torch.nn.functional.one_hot(random_indices, num_classes=num_anchors).float()
                
                # Apply random selection only for masked batches
                # Unsqueeze batch_index_mask to match dimensions for broadcasting
                goal_one_hot = torch.where(batch_index_mask.unsqueeze(-1).unsqueeze(-1), random_one_hot, goal_one_hot)

            # Select anchors using the one-hot tensor
            # Reshape one-hot tensor to [B, A, num_anchors, 1, 1] for broadcasting
            selected_anchors_one_hot = goal_one_hot.unsqueeze(-1).unsqueeze(-1)
            # Multiply and sum to select the target anchors
            target_anchors_gt = (anchors * selected_anchors_one_hot).sum(dim=2)

            anchors_input = torch.diff(target_anchors_gt[:, :, ::2, :], dim=-2)  # Calculate increments from anchors

            # get actions from anchors
            # gt_actions, gt_actions_valid = inverse_kinematics(
            #     anchors_gt,
            #     agents_future_valid,
            #     dt=self._dt,
            #     action_len=self._action_len,
            # )
            # gt_actions_normalized = self.normalize_actions(gt_actions)

            # sample noise
            # noise = torch.randn(B*A, T, D).type_as(agents_future)

            diffusion_steps = torch.randint(
                1, self.noise_scheduler.num_steps * 1 // 20, (B,),
                device=agents_future.device
            ).long().unsqueeze(-1).repeat(1, self._agents_len).view(B, self._agents_len, 1, 1)

            random_diffusion_steps = torch.randint(
                1, self.noise_scheduler.num_steps * 3 // 40, (B,),
                device=agents_future.device
            ).long().unsqueeze(-1).repeat(1, self._agents_len).view(B, self._agents_len, 1, 1) + self.noise_scheduler.num_steps // 40

            diffusion_steps[batch_index_mask] = random_diffusion_steps[batch_index_mask]

            noise = torch.randn(B, self._agents_len, T_future_steps, D_predict).type_as(agents_future)

            # noise the input
            # noised_action_normalized = self.noise_scheduler.add_noise(
            #     gt_actions_normalized,  # .reshape(B*A, T, D),
            #     noise,
            #     diffusion_steps  # , .reshape(B*A),
            # )  # .reshape(B, A, T, D)

            anchors_input = self.normalize_anchor_increments(anchors_input)

            noised_anchors_gt = self.noise_scheduler.add_noise(
                anchors_input,  # .reshape(B*A, T, D),
                noise,
                diffusion_steps  # , .reshape(B*A),
            )
            noised_anchors_gt = torch.clamp(noised_anchors_gt, min=-1, max=1)

            noised_anchors_gt = self.unnormalize_anchor_increments(noised_anchors_gt)

            # if self._replay_buffer:
            #     with torch.no_grad():
            #         # Forward for one step
            #         denoise_outputs = self.forward_denoiser(encoder_outputs, gt_actions_normalized,
            #                                                 diffusion_steps.view(B, A))
            #
            #         x_0 = denoise_outputs['denoised_actions_normalized']
            #
            #         # Step to sample from P(x_t-1 | x_t, x_0)
            #         x_t_prev = self.noise_scheduler.step(
            #             model_output=x_0,
            #             timesteps=diffusion_steps,
            #             sample=noised_action_normalized,
            #             prediction_type=self._prediction_type if hasattr(self, '_prediction_type') else 'sample',
            #         )
            #         noised_action_normalized = x_t_prev.detach()

            # denoise_outputs = self.forward_denoiser(encoder_outputs, noised_action_normalized,
            #                                         diffusion_steps.view(B, A))
            denoise_outputs = self.forward_denoiser(encoder_outputs, noised_anchors_gt,
                                                    diffusion_steps.view(B, self._agents_len))

            debug_outputs.update(denoise_outputs)
            debug_outputs['noise'] = noise
            debug_outputs['diffusion_steps'] = diffusion_steps

            # Get Loss
            denoised_trajs = denoise_outputs['denoised_trajs']
            if self._prediction_type == 'sample':
                # state_loss_mean, yaw_loss_mean = self.denoise_loss(
                #     denoised_trajs,
                #     agents_future, agents_future_valid,
                #     agents_interested,
                # )
                # denoise_loss = state_loss_mean + yaw_loss_mean
                traj_loss = self.traj_loss(
                    denoised_trajs, agents_future, agents_future_valid, agents_interested
                )
                total_loss += self.denoiser_loss_weight * (self.traj_loss_weight * traj_loss)

                # Predict the noise
                # _, diffusion_loss = self.noise_scheduler.get_noise(
                #     x_0=denoise_outputs['denoised_actions_normalized'],
                #     x_t=noised_action_normalized,
                #     timesteps=diffusion_steps,
                #     gt_noise=noise,
                # )
                # _, diffusion_loss = self.noise_scheduler.get_noise(
                #     x_0=denoise_outputs['denoised_trajs_origin'][..., :D_predict],
                #     x_t=noised_anchors_gt,
                #     timesteps=diffusion_steps,
                #     gt_noise=noise,
                # )

                # log_dict.update({
                #     prefix + 'state_loss': state_loss_mean.item(),
                #     prefix + 'yaw_loss': yaw_loss_mean.item(),
                #     prefix + 'diffusion_loss': diffusion_loss.item()
                # })
                log_dict.update({
                    prefix + 'traj_loss': traj_loss.item(),
                    # prefix + 'diffusion_loss': diffusion_loss.item()
                })

            elif self._prediction_type == 'error':
                denoiser_output = denoise_outputs['denoiser_output']
                denoise_loss = torch.nn.functional.mse_loss(
                    denoiser_output, noise, reduction='mean'
                )
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
                    denoised_trajs, agents_future, agents_future_valid, agents_interested
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
                denoised_trajs, agents_future, agents_future_valid, agents_interested, 8
            )

            log_dict.update({
                prefix + 'denoise_ADE': denoise_ade,
                prefix + 'denoise_FDE': denoise_fde,
            })


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
        Calculates the loss for action prediction.

        Args:
            trajs (torch.Tensor): Tensor of shape [B, A, T, 2] representing predicted trajs.
            actions_gt (torch.Tensor): Tensor of shape [B, A, T, 2] representing ground truth trajs.
            actions_valid (torch.Tensor): Tensor of shape [B, A, T] representing validity of trajs.
            agents_interested (torch.Tensor): Tensor of shape [B, A] representing interest in agents.

        Returns:
            action_loss_mean (torch.Tensor): Mean action loss.
        """
        # Get Mask
        trajs_mask = trajs_valid[:, :, 1:] * (agents_interested[..., None] > 0)

        # Calculate the trajs loss
        trajs_loss = smooth_l1_loss(trajs[..., :2], trajs_gt[:, :, 1:, :2], reduction='none').sum(-1)
        trajs_loss = trajs_loss * trajs_mask

        # Calculate the mean loss
        trajs_loss_mean = trajs_loss.sum() / trajs_mask.sum()

        return trajs_loss_mean

    def goal_loss(
            self, trajs, scores, agents_future,
            agents_future_valid, anchors,
            agents_interested
    ):
        """
        Calculates the loss for trajectory prediction using an efficient, vectorized
        implementation of Plackett-Luce ranking loss.

        Args:
            trajs (torch.Tensor): Predicted trajectories from GoalPredictor, shape [B, A, Q, T, 3].
            scores (torch.Tensor): Predicted scores (logits) for anchors, shape [B, A, Q].
            agents_future (torch.Tensor): Ground truth future agent states, shape [B, A, T, D].
            agents_future_valid (torch.Tensor): Validity of future states, shape [B, A, T].
            anchors (torch.Tensor): Anchor trajectories in local frame, shape [B, A, Q, T+1, 2].
            agents_interested (torch.Tensor): Interest in agents, shape [B, A].

        Returns:
            traj_loss_mean (torch.Tensor): Mean regression loss for the best predicted trajectory.
            score_loss_mean (torch.Tensor): Mean ranking loss based on Plackett-Luce model.
        """
        num_batch, num_agents, num_query, _, _ = trajs.shape
        num_timesteps_future = agents_future.shape[2]

        # --- 1. Get Ground Truth Ranking based on Anchors ---
        current_states = agents_future[:, :, 0, :3]
        global_anchors = batch_transform_trajs_to_global_frame(anchors, current_states)

        traj_mask = agents_future_valid[..., 1:] * (agents_interested[..., None] > 0)
        trajs_gt = agents_future[:, :, 1:, :2].flatten(0, 1)
        global_anchors_flat = global_anchors[:, :, :, 1:, :2].flatten(0, 1)
        scores_flat = scores.flatten(0, 1)
        flat_traj_mask = traj_mask.flatten(0, 1)
        flat_agents_interested = agents_interested.flatten(0, 1) > 0

        dist = torch.norm(global_anchors_flat - trajs_gt.unsqueeze(1), dim=-1)
        dist = dist * flat_traj_mask.unsqueeze(1)
        ade = dist.sum(-1) / torch.clamp(flat_traj_mask.sum(-1, keepdim=True), min=1.0)
        gt_ranking = torch.argsort(ade, dim=-1)

        # --- 2. Efficiently Calculate Plackett-Luce ranking loss ---
        # Sort the scores according to the ground truth ranking
        ranked_scores = torch.gather(scores_flat, 1, gt_ranking)

        # Compute log denominators for the Plackett-Luce likelihood
        log_denominators = torch.logcumsumexp(ranked_scores.flip(1), dim=1).flip(1)

        # The log probability of the PL distribution is the sum of the log probabilities
        # of picking the correct item at each step: log(p_k) = r_k - log(sum(exp(r_j)))
        pl_log_probs = ranked_scores - log_denominators
        
        # The total loss is the negative sum of these log probabilities.
        score_loss = -pl_log_probs.sum(dim=-1)

        # Average the loss over valid agents
        score_loss = score_loss * flat_agents_interested
        score_loss_mean = score_loss.sum() / torch.clamp(flat_agents_interested.sum(), min=1.0)

        # --- 3. Calculate regression loss for the best *predicted* trajectory ---
        trajs_pred_flat = trajs.flatten(0, 1)[:, :, :, :2]
        best_pred_idx = torch.argmax(scores_flat, dim=-1)
        trajs_select = trajs_pred_flat[torch.arange(num_batch * num_agents), best_pred_idx]
        
        traj_loss = smooth_l1_loss(trajs_select[:, :num_timesteps_future-1], trajs_gt, reduction='none').sum(-1)
        traj_loss = traj_loss * flat_traj_mask
        traj_loss_mean = traj_loss.sum() / torch.clamp(flat_traj_mask.sum(), min=1.0)

        return traj_loss_mean, score_loss_mean

    def goal_loss_new(
            self, trajs, types, agents_future,
            agents_future_valid, agents_interested
    ):
        """
        Calculates the loss for trajectory prediction.

        Args:
            trajs (torch.Tensor): Tensor of shape [B, A, T, 3] representing predicted trajectories.
            types (torch.Tensor): Tensor of shape [B, A, 10] representing predicted types.
            agents_future (torch.Tensor): Tensor of shape [B, A, T, 3] representing future agent states.
            agents_future_valid (torch.Tensor): Tensor of shape [B, A, T] representing validity of future agent states.
            agents_interested (torch.Tensor): Tensor of shape [B, A] representing interest in agents.

        Returns:
            traj_loss_mean (torch.Tensor): Mean trajectory loss.
            score_loss_mean (torch.Tensor): Mean score loss.
        """

        # Get Mask
        traj_mask = agents_future_valid[..., 1:] * (agents_interested[..., None] > 0)  # [B, A, T]

        # Flatten batch and agents
        trajs_gt = agents_future[:, :, 1:, :3].flatten(0, 1)  # [B*A, T, 3]
        trajs = trajs.flatten(0, 1)[..., :3]  # [B*A, T, 3]
        types = types.flatten(0, 1)  # [B*A, 10]
        types_gt = get_trajectory_type(trajs_gt)

        # Calculate the trajectory loss
        traj_loss = smooth_l1_loss(trajs, trajs_gt, reduction='none').sum(-1)  # [B*A, T]
        traj_loss = traj_loss * traj_mask.flatten(0, 1)  # [B*A, T]

        # Calculate the score loss
        score_loss = cross_entropy(types, types_gt, reduction='none')  # [B*A]
        score_loss = score_loss * (agents_interested.flatten(0, 1) > 0)  # [B*A]

        # Calculate the mean loss
        traj_loss_mean = traj_loss.sum() / traj_mask.sum()
        score_loss_mean = score_loss.sum() / (agents_interested > 0).sum()

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
        goal_trajs = goal_trajs[:, :top_k, :, :, :2]  # [B, A, Q, T, 2]
        gt = agents_future[:, :top_k, 1:, :2]  # [B, A, T, 2]
        gt_mask = (agents_future_valid[:, :top_k, 1:]
                   & (agents_interested[:, :top_k, None] > 0)).bool()  # [B, A, T]

        goal_mse = torch.norm(goal_trajs - gt[:, :, None, :, :], dim=-1)  # [B, A, Q, T]
        goal_mse = goal_mse * gt_mask[..., None, :]  # [B, A, Q, T]
        best_idx = torch.argmin(goal_mse.sum(-1), dim=-1)

        best_goal_mse = goal_mse[torch.arange(goal_mse.shape[0])[:, None],
        torch.arange(goal_mse.shape[1])[None, :],
        best_idx]

        goal_ADE = best_goal_mse.sum() / gt_mask.sum()
        goal_FDE = best_goal_mse[..., -1].sum() / gt_mask[..., -1].sum()

        return goal_ADE.item(), goal_FDE.item()

    @torch.no_grad()
    def calculate_metrics_predict_new(self,
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
        gt_mask = (agents_future_valid[:, :top_k, 1:] \
                   & (agents_interested[:, :top_k, None] > 0)).bool()  # [B, A, T]

        goal_mse = torch.norm(goal_trajs - gt, dim=-1)  # [B, A, T]
        goal_mse = goal_mse * gt_mask  # [B, A, T]
        best_idx = torch.argmin(goal_mse.sum(-1), dim=-1)

        best_goal_mse = goal_mse[torch.arange(goal_mse.shape[0]), best_idx, :]

        goal_ADE = best_goal_mse.sum() / gt_mask.sum()
        goal_FDE = best_goal_mse[..., -1].sum() / gt_mask[..., -1].sum()

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

