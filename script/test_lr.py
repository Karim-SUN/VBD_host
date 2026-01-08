import torch
import math
import matplotlib.pyplot as plt

def get_sgdr_schedule_with_soft_restart(optimizer, warmup_steps, total_steps, 
                                        cycle_len, cycle_mult, decay_rate, min_lr_scale,
                                        cycle_warmup_ratio=0.1): # <--- 新增参数: 周期内热身比例
    """
    创建一个支持 SGDR + Peak Decay + Soft Restart (周期内热身) 的调度器
    """
    def lr_lambda(current_step: int):
        # 1. Global Warmup (全局首个预热阶段)
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        
        # 2. SGDR 阶段
        step_after_warmup = current_step - warmup_steps
        
        current_cycle_len = cycle_len
        cycle_idx = 0
        time_in_cycle = step_after_warmup
        
        # 迭代找到当前 step 处于第几个周期
        while time_in_cycle >= current_cycle_len:
            time_in_cycle -= current_cycle_len
            current_cycle_len = int(current_cycle_len * cycle_mult)
            cycle_idx += 1
        
        # 3. 计算峰值衰减 (Peak Decay)
        peak_scale = decay_rate ** cycle_idx
        
        # --- 4. 周期内逻辑 (Bug 修复版) ---
        
        # [关键修复] 如果是第 0 个周期，直接从最高点开始衰减，跳过 Soft Restart
        # 这样就能完美衔接 Global Warmup 结束时的 1.0
        if cycle_idx == 0:
            # 整个周期都用于衰减
            progress = float(time_in_cycle) / float(max(1, current_cycle_len))
            cosine_value = 0.5 * (1.0 + math.cos(math.pi * progress))
            final_scale = peak_scale * cosine_value
        
        else:
            # Cycle > 0 (重启后) 才启用 Soft Restart
            cycle_warmup_steps = int(current_cycle_len * cycle_warmup_ratio)
            
            if time_in_cycle < cycle_warmup_steps:
                # A. 周期内热身阶段 (Soft Restart)
                # 线性从 0 上升到 peak_scale
                restart_progress = float(time_in_cycle) / float(max(1, cycle_warmup_steps))
                final_scale = peak_scale * restart_progress
                
            else:
                # B. 周期内余弦退火阶段
                # 有效衰减长度 = 总长 - 热身长
                decay_steps = current_cycle_len - cycle_warmup_steps
                time_in_decay = time_in_cycle - cycle_warmup_steps
                
                progress = float(time_in_decay) / float(max(1, decay_steps))
                cosine_value = 0.5 * (1.0 + math.cos(math.pi * progress))
                final_scale = peak_scale * cosine_value
        
        # 5. 最低 LR 保护
        return max(final_scale, min_lr_scale)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def visualize_lr_soft_restart():
    # ================= 配置区域 =================
    BASE_LR = 2e-4
    TOTAL_STEPS = 97420
    WARMUP_STEPS = 8000          # 全局预热 (首个)
    
    # SGDR 参数
    CYCLE_LEN = 20000
    CYCLE_MULT = 1.45
    DECAY_RATE = 0.80
    MIN_LR_SCALE = 0.005
    
    # [新] 周期内热身比例 (0.2 表示每个周期的前 20% 用来热身)
    CYCLE_WARMUP_RATIO = 0.1
    # ===========================================

    print(f"Cycle Warmup Ratio: {CYCLE_WARMUP_RATIO*100}%")

    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=BASE_LR)
    
    scheduler = get_sgdr_schedule_with_soft_restart(
        optimizer, 
        warmup_steps=WARMUP_STEPS, 
        total_steps=TOTAL_STEPS,
        cycle_len=CYCLE_LEN,
        cycle_mult=CYCLE_MULT,
        decay_rate=DECAY_RATE,
        min_lr_scale=MIN_LR_SCALE,
        cycle_warmup_ratio=CYCLE_WARMUP_RATIO # 传入新参数
    )
    
    lrs = []
    steps = []
    
    for step in range(TOTAL_STEPS):
        lrs.append(optimizer.param_groups[0]['lr'])
        steps.append(step)
        scheduler.step()

    # 绘图
    plt.figure(figsize=(12, 6), dpi=100)
    plt.plot(steps, lrs, label='Learning Rate', linewidth=2, color='#2ca02c') # 使用绿色区分
    
    plt.axvline(x=WARMUP_STEPS, color='black', linestyle='--', alpha=0.5, label='Global Warmup End')
    
    # 标注周期
    t = WARMUP_STEPS
    c_len = CYCLE_LEN
    while t < TOTAL_STEPS:
        t += c_len
        if t < TOTAL_STEPS:
            plt.axvline(x=t, color='red', linestyle=':', alpha=0.3)
        c_len = int(c_len * CYCLE_MULT)

    plt.title(f'SGDR with Soft Restart (Cycle Warmup={CYCLE_WARMUP_RATIO*100}%)', fontsize=14)
    plt.xlabel('Steps')
    plt.ylabel('Learning Rate')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    visualize_lr_soft_restart()