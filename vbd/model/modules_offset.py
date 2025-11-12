import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import constant_, xavier_uniform_
from torch.utils.checkpoint import checkpoint
from .model_utils import (batch_transform_trajs_to_local_frame,
                          batch_transform_polylines_to_local_frame,
                          batch_transform_trajs_to_global_frame,
                          roll_out)
from .model_utils_new import get_random_mask


class Encoder(nn.Module):
    def __init__(self, layers=6, version='v1', history_dropout=0.2, task_probabilities=None):
        super().__init__()
        self._version = version
        if self._version == 'v1':
            self.agent_encoder = AgentEncoder()
        elif self._version == 'v2':
            self.agent_encoder = AgentEncoderV2()
        else:
            self.agent_encoder = AgentEncoderTemporalAttn()
        self.map_encoder = MapEncoder()
        self.traffic_light_encoder = TrafficLightEncoder()
        self.relation_encoder = FourierEmbedding(input_dim=3)
        self.transformer_encoder = TransformerEncoder(layers=layers)
        self.history_dropout = history_dropout
        self.task_probabilities = task_probabilities

    def forward(self, inputs):
        # agent encoding
        agents_features = inputs['agents_features']
        B, A_all, T_all, D_all = agents_features.shape
        T_history_and_cur = inputs['T_history_and_cur']
        # x, y, yaw, vx, vy
        agents_interested = inputs['agents_interested'][:, :A_all]
        agents_local = inputs['agents_local'].clone()

        # --- State Dropout: 按时间戳随机丢弃历史状态 ---
        if self.training and self.history_dropout > 0:
            history_len = T_history_and_cur - 1
            if history_len > 0:
                # 创建一个时间步遮罩，形状为 [B, 1, history_len, 1]
                # torch.rand > self.history_dropout 的结果是保留为True(1.0)，丢弃为False(0.0)
                time_mask = (torch.rand(B, 1, history_len, 1, device=agents_local.device) > self.history_dropout).float()
                # 应用遮罩到历史轨迹部分
                agents_local[:, :, :history_len, :] *= time_mask

        multi_task_mask, mask_type = get_random_mask(B, A_all, T_all, T_history_and_cur, task_probabilities=self.task_probabilities)
        agents_mask = torch.eq(agents_interested, 0)
        multi_task_mask[agents_mask.unsqueeze(-1).repeat(1, 1, T_all).bool()] = True

        # 在某些网络结构中（特别是 GRU 或 LSTM），连续的零输入可能会导致隐藏状态（hidden state）趋向于零，这被称为“状态衰减”
        agents_local[multi_task_mask] = 0.0
        agents_local = agents_local.reshape(B * A_all, T_all, D_all)
        mask_type = mask_type.reshape(B * A_all)
        encoded_agents = self.agent_encoder(agents_local, mask_type)
        encoded_agents = encoded_agents.reshape(B, A_all, -1)

        # map and traffic light encoding
        map_polylines = inputs['polylines']
        map_polylines_local = batch_transform_polylines_to_local_frame(map_polylines)
        encoded_map_lanes = self.map_encoder(map_polylines_local)
        maps_mask = inputs['polylines_valid'].logical_not()

        traffic_lights = inputs['traffic_light_points']
        encoded_traffic_lights = self.traffic_light_encoder(traffic_lights)
        traffic_lights_mask = torch.eq(traffic_lights.sum(-1), 0)

        # relation encoding
        relations = inputs['relations']
        relations = self.relation_encoder(relations)

        # transformer encoding
        encoder_outputs = {}
        encoder_outputs['agents'] = agents_features
        encoder_outputs['anchors'] = inputs['anchors']
        encoder_outputs['agents_mask'] = agents_mask
        encoder_outputs['maps_mask'] = maps_mask
        encoder_outputs['traffic_lights_mask'] = traffic_lights_mask
        encoder_outputs['task_mask'] = multi_task_mask
        encoder_outputs['relation_encodings'] = relations
        encoder_outputs['T0'] = T_history_and_cur

        encodings = self.transformer_encoder(relations, encoded_agents, encoded_map_lanes, encoded_traffic_lights,
                                             agents_mask, maps_mask, traffic_lights_mask)
        encoder_outputs['encodings'] = encodings

        return encoder_outputs


class IntentConditioner(nn.Module):
    """
    可控的意图调节器 (替换旧的 GoalPredictor)
    
    任务: 
        输入: 场景 (scene) + 意图 (intent_cluster_idx)
        输出: 粗略的轨迹偏移量 (coarse_offset)
    
    这个模块支持 Classifier-Free Guidance (CFG)，
    通过为 `num_clusters` + 1 (用于 NULL 意图) 来实现。
    """
    def __init__(self, 
                 future_len=80,         # e.g., 80
                 action_len=2,          # e.g., 2 (必须与 VBD_v2.py 和 precompute 匹配)
                 agents_len=32,
                 d_model=256,
                 num_clusters=128,      # 你的簇数量 (e.g., 128)
                 diff_dim=2):           # 你的差分维度 (e.g., 2)
        super().__init__()
        self._agents_len = agents_len
        self.d_model = d_model

        self.num_diff_steps = (future_len // action_len)
        self.diff_dim = diff_dim
        
        # 2. 意图嵌入层 (并行输入)
        # +1 是为 CFG 的 NULL 意图 (e.g., 索引 0) 
        self.cluster_embed = nn.Embedding(num_clusters + 1, d_model)

        # 3. [新] 锚点轨迹先验编码器 (将 [40, 2] 编码为 d_model)
        #    使用一个简单的 MLP 将扁平化的轨迹编码
        self.anchor_traj_encoder = nn.Sequential(
            nn.Linear(self.num_diff_steps * self.diff_dim, d_model * 2),
            nn.ELU(),
            nn.Linear(d_model * 2, d_model)
        )
        self.null_anchor_embedding = nn.Parameter(torch.randn(1, 1, d_model))

        # 4. 融合层 (重用现有的 CrossTransformer)
        self.attention_layers = nn.ModuleList([CrossTransformer() for _ in range(4)])
        
        # 5. 输出头 (回归偏移量)
        self.offset_decoder = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model * 2, d_model * 2),
            nn.ELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model * 2, self.num_diff_steps * self.diff_dim)
        )

        self.score_decoder = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model * 2, d_model * 2),
            nn.ELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model * 2, num_clusters)
        )

    def forward(self, encoder_outputs, intent_command, target_cluster_center_diffs):
        """
        Args:
            encoder_outputs (dict): 来自 Encoder 的输出.
            intent_command (torch.Tensor): 形状为 [B, A_pred] 的意图簇索引 (long).
            target_cluster_center_diffs (torch.Tensor): 形状为 [B, A_pred, T_diff, D_diff] 的簇中心轨迹.
        Returns:
            torch.Tensor: 预测的粗略偏移量, 形状 [B, A_pred, T_diff, D_diff].
        """
        encodings = encoder_outputs['encodings']
        relations = encoder_outputs['relation_encodings']
        mask = torch.cat([encoder_outputs['agents_mask'], 
                          encoder_outputs['maps_mask'],
                          encoder_outputs['traffic_lights_mask']], dim=-1)
        
        num_batch = encodings.shape[0]

        # 1. 将意图索引 [B, A] 转换为嵌入向量
        # 形状: [B, A, D_model]
        intent_embedding = self.cluster_embed(intent_command)

        # 扁平化: [B, A, 80]
        flat_anchor_diffs = target_cluster_center_diffs.flatten(2)
        # 编码: [B, A, D_model]
        anchor_traj_embedding = self.anchor_traj_encoder(flat_anchor_diffs)

        # 当为 NULL 时, 将轨迹嵌入替换为 *可学习* 的 null_anchor_embedding
        null_mask = (intent_command == 0).unsqueeze(-1)
        anchor_traj_embedding = torch.where(
            null_mask, 
            self.null_anchor_embedding, # [B=1, A=1, D] (广播)
            anchor_traj_embedding        # [B, A, D]
        )
        
        # 2. 为 CrossTransformer 准备查询 (Query)
        # 形状: [B, A, 1, D_model]
        query_base = intent_embedding + anchor_traj_embedding
        query = query_base.unsqueeze(2)

        coarse_offsets_list = []
        cluster_score_list = []
        # 3. 循环遍历每个 agent
        for i in range(self._agents_len):
            # 形状: [B, 1, D_model]
            query_agent_intent = query[:, i]

            # 4. 融合: (意图) 查询 (场景)
            query_content = self.attention_layers[0](query_agent_intent, encodings, relations[:, i], key_mask=mask)
            query_content = self.attention_layers[1](query_content, encodings, relations[:, i], key_mask=mask)
            query_content = query_content + query_agent_intent
            query_content = self.attention_layers[2](query_content, encodings, relations[:, i], key_mask=mask)
            query_content = self.attention_layers[3](query_content, encodings, relations[:, i], key_mask=mask)
            
            # 5. 解码
            # 形状: [B, 1, D_model] -> [B, D_model]
            final_context_intent = query_content.squeeze(1)
            
            # 形状: [B, D_model] -> [B, T_diff * D_diff]
            coarse_offset = self.offset_decoder(final_context_intent)
            
            # 形状: [B, T_diff, D_diff]
            coarse_offset = coarse_offset.view(num_batch, self.num_diff_steps, self.diff_dim)
            coarse_offsets_list.append(coarse_offset)

            cluster_score = self.score_decoder(final_context_intent)
            cluster_score_list.append(cluster_score)

        # 6. 堆叠所有 agent 的结果
        # 形状: [B, A_pred, T_diff, D_diff]
        all_coarse_offsets = torch.stack(coarse_offsets_list, dim=1)
        all_cluster_scores = torch.stack(cluster_score_list, dim=1)

        return all_coarse_offsets, all_cluster_scores

    def reset_agent_length(self, agents_len):
        self._agents_len = agents_len


class Denoiser(nn.Module):
    def __init__(self, future_len=80, action_len=5, agents_len=32, steps=100, input_dim=5):
        super().__init__()
        self._agents_len = agents_len
        self._action_len = action_len
        self._input_dim = input_dim
        self.noise_level_embedding = nn.Embedding(steps, 256)
        self.decoder = TransformerDecoder(future_len, agents_len, self._action_len, input_dim=self._input_dim)

    def forward(self, encoder_inputs, noised_trajs, diffusion_step, rollout=True):
        '''
        Args:
            noised_target_offset: [B, A, T_r, 2], [acc, yaw_rate] Unnormalized actions
            diffusion_step: [B, A]
        Output:
            denoised_output: [B, A, T_r, 2], [x, y]
        '''
        noised_trajs = noised_trajs[:, :self._agents_len]

        if type(diffusion_step) == int:
            diffusion_step = torch.full(
                noised_trajs.shape[:-2], diffusion_step,
                dtype=torch.long, device=noised_trajs.device
            )
        else:
            diffusion_step = diffusion_step[:, :self._agents_len]

        T_history_and_cur = encoder_inputs['T0']
        current_states = encoder_inputs['agents'][:, :self._agents_len, T_history_and_cur - 1]

        encodings = encoder_inputs['encodings']
        relations = encoder_inputs['relation_encodings']

        agents_mask = encoder_inputs['agents_mask']
        maps_mask = encoder_inputs['maps_mask']
        traffic_lights_mask = encoder_inputs['traffic_lights_mask']
        mask = torch.cat([agents_mask, maps_mask, traffic_lights_mask], dim=-1)

        # denoise step
        noise_level = self.noise_level_embedding(diffusion_step)
        if rollout:
            embedding = roll_out(current_states, noised_trajs,
                                 action_len=self._action_len, global_frame=False)
        else:
            embedding = noised_trajs

        decoder_output = self.decoder(
            embedding, noise_level,
            encodings, relations, mask
        )

        # # Decoder learns the offset
        # decoded_trajs = noised_trajs + decoder_output

        # return decoded_trajs

        # Decoder learns the target offset
        return decoder_output

    def reset_agent_length(self, agents_len):
        self._agents_len = agents_len
        self.decoder.reset_agent_length(agents_len)


class AgentEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.motion = nn.GRU(5, 256, 2, batch_first=True, dropout=0.2)  # Add dropout
        self.type_embed = nn.Embedding(5, 256, padding_idx=0)

    def forward(self, history, type):
        traj, _ = self.motion(history)
        output = traj[:, -1]  # current frame
        type_embed = self.type_embed(type)
        output = output + type_embed

        return output


class AgentEncoderV2(nn.Module):
    def __init__(self):
        super().__init__()
        self.motion = nn.Sequential(
            nn.Linear(5, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
        )
        self.type_embed = nn.Embedding(5, 256, padding_idx=0)

    def forward(self, history, type):
        cur = history[:, -1, :]
        output = self.motion(cur)
        type_embed = self.type_embed(type)
        output = output + type_embed

        return output


class AgentEncoderTemporalAttn(nn.Module):
    def __init__(self, time_dim=5, embed_dim=256, heads=4, max_len=100):
        super().__init__()
        self.embed = nn.Linear(time_dim, embed_dim)
        self.pos = nn.Parameter(torch.randn(1, max_len, embed_dim))
        self.mha = nn.MultiheadAttention(embed_dim, heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        self.type_embed = nn.Embedding(5, embed_dim, padding_idx=0)

    def forward(self, history, type):
        # history: [B*A, T, 5]
        x = self.embed(history)  # [B*A, T, D]
        T = x.shape[1]
        pos = self.pos[:, :T, :]
        x = x + pos
        x2, _ = self.mha(x, x, x)  # self-attention over time
        x = self.norm(x + x2)
        out = torch.max(x, dim=1).values  # Aggregate by max-pooling over time
        out = out + self.type_embed(type)
        return out


class MapEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.point = nn.Sequential(nn.Linear(3, 128), nn.ReLU(), nn.Linear(128, 256))
        self.traffic_light_embed = nn.Embedding(8, 256)
        self.type_embed = nn.Embedding(21, 256, padding_idx=0)

    def forward(self, inputs):
        # inputs [B, M, W, 5]
        output = self.point(inputs[..., :3])
        output = torch.max(output, dim=-2).values  # max pooling on W

        traffic_light_type = inputs[:, :, 0, 3].long().clamp(0, 7)
        traffic_light_embed = self.traffic_light_embed(traffic_light_type)
        polyline_type = inputs[:, :, 0, 4].long().clamp(0, 20)
        type_embed = self.type_embed(polyline_type)
        output = output + traffic_light_embed + type_embed

        return output


class TrafficLightEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.type_embed = nn.Embedding(8, 256)

    def forward(self, inputs):
        # inputs [B, TL, 3]
        traffic_light_type = inputs[:, :, 2].long().clamp(0, 7)
        type_embed = self.type_embed(traffic_light_type)
        output = type_embed

        return output


class QCMHA(nn.Module):
    """
    Quadratic Complexity Multi-Head Attention module.

    Args:
        embed_dim (int): The dimension of the input embeddings.
        num_heads (int): The number of attention heads.
        dropout (float, optional): The dropout probability. Default is 0.1.
    """

    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        self.in_proj = nn.Linear(embed_dim, 3 * embed_dim, bias=True)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)

        self.dropout = nn.Dropout(dropout)

        self._reset_parameters()

    def _reset_parameters(self):
        xavier_uniform_(self.in_proj.weight)
        xavier_uniform_(self.out_proj.weight)
        constant_(self.in_proj.bias, 0.)
        constant_(self.out_proj.bias, 0.)

    def forward(self, query, rel_pos, attn_mask=None):
        '''
        Forward pass of the QCMHA module.

        Args:
            query (torch.Tensor): The input query tensor of shape [batch_size, query_length, embed_dim].
            rel_pos (torch.Tensor): The relative position tensor of shape [batch_size, query_length, key_length, embed_dim].
            attn_mask (torch.Tensor, optional): The attention mask tensor of shape [batch_size, query_length, key_length].

        Returns:
            torch.Tensor: The output tensor of shape [batch_size, query_length, embed_dim].
        '''
        query = self.in_proj(query)
        b, t, d = query.shape
        query = query.reshape(b, t, self.num_heads, self.head_dim * 3)

        res = torch.split(query, self.head_dim, dim=-1)
        q, k, v = res

        rel_pos_q = rel_pos_v = rel_pos

        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 3, 1)
        v = v.permute(0, 2, 1, 3)

        dot_score = torch.matmul(q, k)

        if rel_pos is not None:
            rel_pos_q = rel_pos_q.reshape(b, t, t, self.num_heads, self.head_dim)
            rel_pos_q = rel_pos_q.permute(0, 3, 1, 4, 2)  # [b, h, q, d, k]
            # [b, h, q, 1, d] * [b, h, q, d, k] -> [b, h, q, 1, k]
            dot_score_rel = torch.matmul(q.unsqueeze(-2), rel_pos_q).squeeze(-2)
            dot_score += dot_score_rel

        dot_score = dot_score / np.sqrt(self.head_dim)

        if attn_mask is not None:
            dot_score = dot_score - attn_mask.float() * 1e9

        dot_score = F.softmax(dot_score, dim=-1)
        dot_score = self.dropout(dot_score)

        value = torch.matmul(dot_score, v)

        if rel_pos is not None:
            rel_pos_v = rel_pos_v.reshape(b, t, t, self.num_heads, self.head_dim)
            rel_pos_v = rel_pos_v.permute(0, 3, 1, 2, 4)  # [b, h, q, k, d]
            # [b, h, q, 1, k] * [b, h, q, k, d] -> [b, h, q, d]
            value_rel = torch.matmul(dot_score.unsqueeze(-2), rel_pos_v).squeeze(-2)
            value += value_rel

        value = value.permute(0, 2, 1, 3)  # [b, t, h, d//h]
        value = value.reshape(b, t, self.embed_dim)
        value = self.out_proj(value)

        return value


class SelfTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        heads, dim, dropout = 8, 256, 0.1
        self.qc_attention = QCMHA(dim, heads, dropout)
        self.norm_1 = nn.LayerNorm(dim)
        self.norm_2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(dim * 4, dim), nn.Dropout(dropout))

    def forward(self, inputs, relations, mask=None):
        # attention_output = self.qc_attention(inputs, relations, mask)
        # attention_output = self.norm_1(attention_output + inputs)
        # output = self.norm_2(self.ffn(attention_output) + attention_output)
        attention_output = checkpoint(self.qc_attention, inputs, relations, mask, use_reentrant=False)
        attention_output = self.norm_1(attention_output + inputs)
        ffn_output = checkpoint(self.ffn, attention_output, use_reentrant=False)
        output = self.norm_2(ffn_output + attention_output)

        return output


class FourierEmbedding(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, num_freq_bands=64):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.freqs = nn.Embedding(input_dim, num_freq_bands) if input_dim != 0 else None

        self.mlps = nn.ModuleList(
            [nn.Sequential(
                nn.Linear(num_freq_bands * 2 + 1, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, hidden_dim),
            ) for _ in range(input_dim)])

        self.to_out = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, continuous_inputs):
        x = continuous_inputs.unsqueeze(-1) * self.freqs.weight * 2 * math.pi
        x = torch.cat([x.cos(), x.sin(), continuous_inputs.unsqueeze(-1)], dim=-1)
        x = torch.stack([self.mlps[i](x[:, :, :, i]) for i in range(self.input_dim)]).sum(dim=0)

        return self.to_out(x)


class TransformerEncoder(nn.Module):
    def __init__(self, layers=6):
        super().__init__()
        self.layers = nn.ModuleList([SelfTransformer() for _ in range(layers)])

    def forward(self, encoded_relations, encoded_trajs, encoded_polylines, encoded_traffic_lights,
                trajs_mask, polylines_mask, traffic_lights_mask):
        # relations: [B, N+M+TL, N+M+TL, 256]
        # encoded_trajs: [B, N, 256]
        # encoded_polylines: [B, M, 256]
        # encoded_traffic_lights: [B, TL, 256]

        encodings = torch.cat([encoded_trajs, encoded_polylines, encoded_traffic_lights], dim=1)
        encodings_mask = torch.cat([trajs_mask, polylines_mask, traffic_lights_mask], dim=-1)
        attention_mask = encodings_mask.unsqueeze(-1).repeat(1, 1, encodings_mask.shape[1])
        attention_mask = attention_mask.unsqueeze(1)

        for layer in self.layers:
            encodings = layer(encodings, encoded_relations, attention_mask)

        return encodings


class CrossTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        heads, dim, dropout = 8, 256, 0.1
        self.cross_attention = nn.MultiheadAttention(dim, heads, dropout, batch_first=True)
        self.norm_1 = nn.LayerNorm(dim)
        self.norm_2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(dim * 4, dim), nn.Dropout(dropout))

    def forward(self, query, key, relations, attn_mask=None, key_mask=None):
        # add relations to key and value
        key = key + relations
        value = key

        # if key_mask is not None:
        #     attention_output, _ = self.cross_attention(query, key, value, key_padding_mask=key_mask)
        # elif attn_mask is not None:
        #     attention_output, _ = self.cross_attention(query, key, value, attn_mask=attn_mask)
        # else:
        #     attention_output, _ = self.cross_attention(query, key, value)

        # attention_output = self.norm_1(attention_output)
        # output = self.norm_2(self.ffn(attention_output) + attention_output)

        def _run_attention(cross_attn_module, query, key, value, attn_mask, key_mask):
            if key_mask is not None:
                return cross_attn_module(query, key, value, key_padding_mask=key_mask)[0]
            elif attn_mask is not None:
                return cross_attn_module(query, key, value, attn_mask=attn_mask)[0]
            else:
                return cross_attn_module(query, key, value)[0]

        attention_output = checkpoint(
            _run_attention,
            self.cross_attention,
            query,
            key,
            value,
            attn_mask,
            key_mask,
            use_reentrant=False
        )
        
        attention_output = self.norm_1(attention_output)

        ffn_output = checkpoint(
            self.ffn, 
            attention_output, 
            use_reentrant=False
        )
        output = self.norm_2(ffn_output + attention_output)

        return output


class TransformerDecoder(nn.Module):
    def __init__(self, future_len, agents_len, action_len, input_dim=5, ouptut_dim=2, causal=True):
        super().__init__()
        self._future_len = future_len
        self._action_len = action_len
        self._agents_len = agents_len
        self._future_len = future_len // action_len
        self._input_dim = input_dim
        self._output_dim = ouptut_dim

        self.time_embedding = nn.Embedding(self._future_len, 256)
        self.attention_layers = nn.ModuleList([CrossTransformer() for _ in range(4)])
        self.encoder = nn.Sequential(nn.Linear(self._input_dim, 128), nn.ReLU(), nn.Linear(128, 256))
        self.decoder = nn.Sequential(nn.Linear(256, 128), nn.ELU(), nn.Dropout(0.1), nn.Linear(128, self._output_dim))

        self.register_buffer('casual_mask', self.generate_casual_mask(causal))
        self.register_buffer('time', torch.arange(self._future_len).unsqueeze(0))

    def generate_casual_mask(self, causal=True):
        if not causal:
            return torch.zeros(self._agents_len, self._future_len, self._agents_len * self._future_len, dtype=bool)

        # Initialize a zero mask
        mask = torch.zeros(self._agents_len, self._future_len, self._agents_len * self._future_len)

        # An agent can attend to all of its own actions
        for i in range(self._agents_len):
            mask[i, :, i * self._future_len:(i + 1) * self._future_len] = 1.0

        # An agent can attend to other agents from all previous timesteps but not future timesteps
        for i in range(self._agents_len):
            for j in range(self._agents_len):
                if i != j:
                    for t in range(self._future_len):
                        mask[i, t, j * self._future_len:j * self._future_len + t + 1] = 1.0

        # Convert to boolean mask
        mask = mask.bool().logical_not()

        return mask

    def forward(self, noisy_trajectories, noise_level, encodings, relations, mask):
        '''
        noisy_trajectories: [B, Na, T_f, 5]
        '''
        # get query
        future_states = self.encoder(noisy_trajectories) # [B, Na, T, 256]
        time_embedding = self.time_embedding(self.time)  # [1, T, 256]
        query = future_states + time_embedding[:, None]  # [B, Na, T, 256]
        query = query + noise_level[:, :, None, :]

        # decode denoised actions
        query_content_list = []
        for i in range(self._agents_len):
            query_content = self.attention_layers[0](
                query[:, i],
                query.reshape(-1, self._agents_len * self._future_len, 256),
                relations[:, i, :self._agents_len].repeat_interleave(self._future_len, dim=1),
                attn_mask=self.casual_mask[i])  # [B, T, 256]
            query_content = self.attention_layers[1](query_content, encodings, relations[:, i],
                                                     key_mask=mask)  # [B, T, 256]
            query_content_list.append(query_content)

        query_content_stack = torch.stack(query_content_list, dim=1)  # [B, Na, T, 256]
        query_content_stack = query_content_stack + query

        query_content_list = []
        for i in range(self._agents_len):
            query_content = self.attention_layers[2](
                query_content_stack[:, i],
                query_content_stack.reshape(-1, self._agents_len * self._future_len, 256),
                relations[:, i, :self._agents_len].repeat_interleave(self._future_len, dim=1),
                attn_mask=self.casual_mask[i])  # [B, T, 256]
            query_content = self.attention_layers[3](query_content, encodings, relations[:, i],
                                                     key_mask=mask)  # [B, T, 256]
            query_content_list.append(query_content)

        query_content_stack = torch.stack(query_content_list, dim=1)  # [B, Na, T, 256]
        actions = self.decoder(query_content_stack)

        return actions

    def reset_agent_length(self, agents_len):
        self._agents_len = agents_len
        new_mask = self.generate_casual_mask().type_as(self.casual_mask)
        self.casual_mask = new_mask