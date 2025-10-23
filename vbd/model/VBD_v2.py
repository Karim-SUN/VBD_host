import numpy as np
import torch
import bitsandbytes.optim as bnb_optim
import lightning.pytorch as pl
from .modules_v2 import Encoder, Denoiser, GoalPredictor
from .utils import DDPM_Sampler
from .model_utils_new import (inverse_kinematics, roll_out, 
                              batch_transform_trajs_to_global_frame,
                              batch_transform_trajs_to_local_frame,
                              get_trajectory_type, interpolate_anchors, roll_out_new)
from torch.nn.functional import smooth_l1_loss, cross_entropy, gumbel_softmax, binary_cross_entropy_with_logits
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
        self.use_gumbel_anneal = cfg.get('use_gumbel_anneal', False)
        self._gumbel_tau_start = cfg.get('gumbel_tau_start', 1.0)
        self._gumbel_tau_end = cfg.get('gumbel_tau_end', 0.1)
        self._gumbel_anneal_steps = cfg.get('gumbel_anneal_steps', 10000)

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

        optimizer = bnb_optim.AdamW8bit(  # 使用 8-bit 版本
            params_to_update,
            lr=self.cfg['lr'],
            weight_decay=self.cfg['weight_decay']
        )

        warmup_steps = self.cfg['lr_warmup_step'] // self.accumulate_grad_batches
        total_steps = self.cfg['lr_total_steps'] // self.accumulate_grad_batches
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
        denoised_offset = self.noise_scheduler.q_x0(
            denoiser_output,
            diffusion_step,
            noised_inputs,
            prediction_type=self._prediction_type
        )
        T_history_and_cur = encoder_outputs['T0']
        current_states = encoder_outputs['agents'][:, :self._agents_len, T_history_and_cur - 1]
        assert encoder_outputs['agents'].shape[1] >= self._agents_len, 'Too many agents to consider'

        # Roll out
        # When using decoder to predict the offset, the denoised_trajs is the original anchors + offset
        final_offset = denoised_offset + anchor_diff
        denoised_trajs, denoised_trajs_origin = roll_out_new(
            current_states, final_offset, global_frame=True)

        return {
            'denoiser_output': denoiser_output,
            'denoised_offset': denoised_offset,
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
        B, A_all, T_future_and_cur, D_all = agents_future.shape
        T_future_steps = T_future_and_cur // self._action_len
        D_predict = 2
        # batch['anchors']: B, A_pred, Q, T_future_and_cur, D_predict
        batch['anchors'] = self.anchor_tensor.unsqueeze(0).unsqueeze(0).expand(B, self._agents_len, -1, -1, -1)

        # TODO: Investigate why this to NAN
        # agents_future_valid = batch['agents_future_valid'][:, :self._agents_len]
        agents_future_valid = torch.ne(agents_future.sum(-1), 0)
        agents_future_valid = agents_future_valid[:, :, 1].unsqueeze(-1).expand_as(
            agents_future_valid) & agents_future_valid # B, A_all, T_future_and_cur
        agents_interested = batch['agents_interested']
        anchors = batch['anchors']
        
        # --- 数据拼接：将历史和未来轨迹拼接为 agent 特征 ---
        agents_history = batch['agents_history']
        agents_features = torch.cat((agents_history[:, :, :-1, :5], agents_future[..., :5]), dim=-2)
        batch['agents_features'] = agents_features # B, A_all, T_history_and_cur + T_future_and_cur - 1, 5
        T_history_and_cur = agents_history.shape[-2]
        batch['T_history_and_cur'] = T_history_and_cur
        batch['T_history_and_cur'] = agents_history.shape[-2]

        log_dict = {}
        debug_outputs = {}
        total_loss = 0

        goal_scores = None


        ############## Run Encoder ##############
        encoder_outputs = self.encoder(batch)
        agents_future = agents_future[:, :self._agents_len] # B, A_pred, T_future_and_cur, 5
        agents_future_valid = agents_future_valid[:, :self._agents_len] # B, A_pred, T_future_and_cur
        agents_interested = agents_interested[:, :self._agents_len] # B, A_pred


        ############### Behavior Prior Prediction #################
        if self._train_predictor:
            goal_outputs = self.forward_predictor(encoder_outputs)
            debug_outputs.update(goal_outputs)

            # get loss
            goal_scores = goal_outputs['goal_scores'] # B, A_pred, Q
            # goal_types = goal_outputs['goal_types']
            goal_trajs = goal_outputs['goal_trajs'] # B, A_pred, Q, T_future, 5

            (goal_loss_mean, 
            score_loss_mean, 
            gt_ranking, # B * A_pred, Q 
            ade_for_ranking # B * A_pred, Q
            ) = self.goal_loss(
                goal_trajs, goal_scores, agents_future,
                agents_future_valid, anchors,
                agents_interested,
            )

            pred_loss = self.goal_loss_weight * goal_loss_mean + self.score_loss_weight * score_loss_mean
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


        ############### Denoise #################
        if self._train_denoiser:
            # get predicted anchor
            assert goal_scores != None, 'No valid goal predictions yet.'
            B_idx = torch.arange(B).unsqueeze(1)  # 生成形状为 [B, 1] 的批次索引
            A_idx = torch.arange(self._agents_len).unsqueeze(0)  # 生成形状为 [1, A_pred] 的车辆索引

            # B, A_all, T_history_and_cur + T_future_and_cur - 1, 5
            agents_local = batch_transform_trajs_to_local_frame(agents_features, ref_idx=T_history_and_cur - 1)
            # B, A_pred, T_future_and_cur, 2
            gt_future_local = agents_local[:, :self._agents_len, T_history_and_cur - 1:, :2]
            # B, A_pred, T_future_steps, 2
            gt_future_diff = torch.diff(gt_future_local[:, :, ::2, :], dim=-2)

            # Use best matched anchor to train
            best_anchor_idx = gt_ranking[:, 0].view(B, self._agents_len)  # B, A_pred

            all_anchor_diff = torch.diff(anchors[:, :, :, ::2, :], dim=-2)  # B, A_pred, Q, T_future_steps, 2

            best_anchor_diff = all_anchor_diff[
                B_idx, A_idx, best_anchor_idx
            ]  # B, A_pred, T_future_steps, 2
            target_offset = gt_future_diff - best_anchor_diff  # B, A_pred, T_future_steps, 2

            # Predicted anchor
            best_pred_anchor_idx = goal_scores.argmax(dim=-1)  # B, A_pred
            best_pred_anchor_diff = all_anchor_diff[
                B_idx, A_idx, best_pred_anchor_idx
            ]  # B, A_pred, T_future_steps, 2

            # # Use Gumbel-Softmax for differentiable sampling of anchors
            # goal_one_hot = gumbel_softmax(goal_scores, tau=self._gumbel_tau, hard=True, dim=-1)

            # # Introduce random exploration
            # batch_index_mask = torch.rand(B, device=goal_scores.device) < self._random_target
            # if batch_index_mask.any():
            #     num_anchors = goal_scores.shape[-1]
            #     random_indices = torch.randint(0, num_anchors, (B, self._agents_len), device=goal_scores.device)
            #     random_one_hot = torch.nn.functional.one_hot(random_indices, num_classes=num_anchors).float()
                
            #     # Apply random selection only for masked batches
            #     # Unsqueeze batch_index_mask to match dimensions for broadcasting
            #     goal_one_hot = torch.where(batch_index_mask.unsqueeze(-1).unsqueeze(-1), random_one_hot, goal_one_hot)

            # # Select anchors using the one-hot tensor
            # # Reshape one-hot tensor to [B, A, num_anchors, 1, 1] for broadcasting
            # selected_anchors_one_hot = goal_one_hot.unsqueeze(-1).unsqueeze(-1)
            # # Multiply and sum to select the target anchors
            # target_anchors_gt = (anchors * selected_anchors_one_hot).sum(dim=2)

            # anchors_input = torch.diff(target_anchors_gt[:, :, ::2, :], dim=-2)  # Calculate increments from anchors

            diffusion_steps = torch.randint(
                1, self.noise_scheduler.num_steps * 1 // 20, (B,),
                device=agents_future.device
            ).long().unsqueeze(-1).repeat(1, self._agents_len).view(B, self._agents_len, 1, 1) # B, A_pred, 1, 1

            # random_diffusion_steps = torch.randint(
            #     1, self.noise_scheduler.num_steps * 3 // 40, (B,),
            #     device=agents_future.device
            # ).long().unsqueeze(-1).repeat(1, self._agents_len).view(B, self._agents_len, 1, 1) + self.noise_scheduler.num_steps // 40

            # diffusion_steps[batch_index_mask] = random_diffusion_steps[batch_index_mask]

            # noise = torch.randn(B, self._agents_len, T_future_steps, D_predict).type_as(agents_future)

            noise = torch.randn_like(target_offset) # B, A_pred, T_future_steps, 2

            target_offset_norm = self.normalize_anchor_increments(target_offset)

            noised_target_offset_norm = self.noise_scheduler.add_noise(
                target_offset_norm,
                noise,
                diffusion_steps
            )
            noised_target_offset_norm = torch.clamp(noised_target_offset_norm, min=-1, max=1)

            noised_target_offset = self.unnormalize_anchor_increments(noised_target_offset_norm) # B, A_pred, T_future_steps, 2

            # Inverse diffusion
            denoise_outputs = self.forward_denoiser(encoder_outputs, noised_target_offset,
                                                    diffusion_steps.view(B, self._agents_len), 
                                                    best_pred_anchor_diff)
            # denoise_outputs['denoiser_output']: B, A_pred, T_future_steps, 2
            # denoise_outputs['denoised_offset']: B, A_pred, T_future_steps, 2
            # denoise_outputs['denoised_trajs']: B, A_pred, T_future, 5
            # denoise_outputs['denoised_trajs_origin']: B, A_pred, T_future, 5

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

        log_dict['gumbel_tau'] = self._gumbel_tau
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

        if self.use_gumbel_anneal:
            if global_step >= self._gumbel_anneal_steps // self.accumulate_grad_batches:
                self._gumbel_tau = self._gumbel_tau_end
            else:
                tau_decay = (self._gumbel_tau_start - self._gumbel_tau_end) * (
                    1 - global_step / (self._gumbel_anneal_steps // self.accumulate_grad_batches)
                )
                self._gumbel_tau = self._gumbel_tau_end + tau_decay
        else:
            self._gumbel_tau = self._gumbel_tau_end

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

    def goal_loss(
            self, trajs, scores, agents_future,
            agents_future_valid, anchors,
            agents_interested
    ):
        """
        Calculates the loss for trajectory prediction.
        Supports BCE, Plackett-Luce (rank), and a mix of both for the score loss.

        Args:
            trajs (torch.Tensor): Predicted trajectories from GoalPredictor, shape [B, A_pred, Q, T_future, 5].
            scores (torch.Tensor): Predicted scores (logits) for anchors, shape [B, A_pred, Q].
            agents_future (torch.Tensor): Ground truth future agent states, shape [B, A_pred, T_future_and_cur, 5].
            agents_future_valid (torch.Tensor): Validity of future states, shape [B, A_pred, T_future_and_cur].
            anchors (torch.Tensor): Anchor trajectories in local frame, shape [B, A_pred, Q, T_future_and_cur, 2].
            agents_interested (torch.Tensor): Interest in agents, shape [B, A_pred].

        Returns:
            traj_loss_mean (torch.Tensor): Mean regression loss for the best predicted trajectory.
            score_loss_mean (torch.Tensor): Mean score loss.
            gt_ranking (torch.Tensor): Ground truth ranking of anchors based on ADE, shape [B * A_pred, Q].
            ade (torch.Tensor): ADE for each anchor, shape [B * A_pred, Q].
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

        # --- 2. Calculate Score Loss based on configured type ---
        score_loss = 0.0

        # BCE Loss Calculation
        if self.score_loss_type in ['bce', 'mix']:
            best_anchor_idx = gt_ranking[:, 0]
            bce_target = torch.nn.functional.one_hot(best_anchor_idx, num_classes=num_query).float()
            bce_loss = binary_cross_entropy_with_logits(scores_flat, bce_target, reduction='none').sum(dim=-1)
            score_loss += self.bce_loss_weight * bce_loss

        # Rank Loss (Plackett-Luce) Calculation
        if self.score_loss_type in ['rank', 'mix']:
            ranked_scores = torch.gather(scores_flat, 1, gt_ranking)
            log_denominators = torch.logcumsumexp(ranked_scores.flip(1), dim=1).flip(1)
            pl_log_probs = ranked_scores - log_denominators

            if self.use_focused_rank_loss:
                rank_topk_mask = torch.zeros_like(pl_log_probs)
                rank_topk_mask[:, :self.focused_rank_topk] = 1.0
                pl_log_probs = pl_log_probs * rank_topk_mask

            rank_loss = -pl_log_probs.sum(dim=-1)

            # Soft Contrastive Hinge Loss
            if self.use_hinge_loss:
                worst_topk_idx = gt_ranking[:, self.focused_rank_topk - 1]
                best_bottomk_idx = gt_ranking[:, -self.focused_rank_topk]
                worst_topk_scores = torch.gather(scores_flat, 1, worst_topk_idx.unsqueeze(-1)).squeeze(-1)
                best_bottomk_scores = torch.gather(scores_flat, 1, best_bottomk_idx.unsqueeze(-1)).squeeze(-1)
                score_diff_logits = worst_topk_scores - best_bottomk_scores
                target = torch.ones_like(score_diff_logits)
                soft_hinge_loss = binary_cross_entropy_with_logits(
                    score_diff_logits, target, reduction='none').sum(dim=-1)
                rank_loss += self.soft_hinge_loss_weight * soft_hinge_loss

            score_loss += self.rank_loss_weight * rank_loss

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

        return traj_loss_mean, score_loss_mean, gt_ranking, ade

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

