"""
KD Diagnostic Script — Forward-pass-only experiments to debug KD issues.
Run with: torchrun --standalone --nproc_per_node=1 kd_diagnostic.py
"""
import os
import sys
import torch
import torch.nn.functional as F
import torch.distributed as dist
import numpy as np

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

# Force single GPU
torch.empty(1, device=f"cuda:{os.environ['LOCAL_RANK']}", requires_grad=True).backward()

rank = int(os.environ["RANK"])
world_size = int(os.environ["WORLD_SIZE"])
device = torch.device(f"cuda:{os.environ['LOCAL_RANK']}")

# Import from train_gpt (this works because torchrun sets up the env)
from train_gpt import (
    GPT, Block, AttnArgs, ForwardScheduleConfig, CastedLinearT,
    distributed_data_generator, TRAINING_STAGES, Hyperparameters,
    norm, next_multiple_of_n
)
from teacher_model import StudentTeacherWrapper, load_student_as_teacher

args = Hyperparameters()
grad_accum_steps = 8 // world_size

print("=" * 60)
print("KD DIAGNOSTIC SCRIPT")
print("=" * 60)

TEACHER_CKPT = os.environ.get("TEACHER_CHECKPOINT", "checkpoints/ce-baseline/state_step001530.pt")
print(f"Teacher checkpoint: {TEACHER_CKPT}")

# ============================================================
# Load teacher as proper GPT (not wrapper)
# ============================================================
print("\n--- Loading teacher as GPT (proper forward, with bigrams) ---")
ckpt = torch.load(TEACHER_CKPT, map_location=device, weights_only=False)
state = ckpt['model']
state = {k.removeprefix('_orig_mod.'): v for k, v in state.items()}

max_seq_len = args.val_batch_size // (grad_accum_steps * world_size)
teacher_gpt = GPT(
    vocab_size=50257, num_layers=11, num_heads=6, head_dim=128, model_dim=768,
    max_seq_len=max_seq_len, kd_mode=False
)
missing, unexpected = teacher_gpt.load_state_dict(state, strict=False)
print(f"  Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")
if missing:
    print(f"  Missing: {missing[:5]}...")
if unexpected:
    print(f"  Unexpected: {unexpected[:5]}...")

for m in teacher_gpt.modules():
    if isinstance(m, (torch.nn.Embedding, torch.nn.Linear)):
        m.weight.data = m.weight.data.bfloat16()
teacher_gpt.attn_gate_bank.data = teacher_gpt.attn_gate_bank.data.bfloat16()
teacher_gpt.ve_gate_bank.data = teacher_gpt.ve_gate_bank.data.bfloat16()
teacher_gpt.attn_bank.data = teacher_gpt.attn_bank.data.bfloat16()
teacher_gpt.mlp_bank.data = teacher_gpt.mlp_bank.data.bfloat16()
teacher_gpt.cuda()
teacher_gpt.eval()
teacher_gpt.requires_grad_(False)

# Also load via wrapper for comparison
print("\n--- Loading teacher via StudentTeacherWrapper (zeroed bigrams) ---")
teacher_wrapper = load_student_as_teacher(
    TEACHER_CKPT, device, max_seq_len, GPT, attn_args_class=AttnArgs
)

# ============================================================
# EXPERIMENT A: Teacher quality comparison
# ============================================================
print("\n" + "=" * 60)
print("EXPERIMENT A: Teacher val_loss — proper GPT vs wrapper")
print("=" * 60)

val_loader = distributed_data_generator(
    args.val_files, args.val_batch_size, -1,
    grad_accum_steps=grad_accum_steps, align_to_bos=False
)
val_steps = grad_accum_steps * args.val_tokens // args.val_batch_size

# We need a schedule config for the proper GPT forward
w0 = TRAINING_STAGES[0]

loss_gpt = 0.0
loss_wrapper = 0.0
with torch.no_grad():
    for i in range(val_steps):
        inputs, targets, cum_seqlens, bigram_inputs, _ = next(val_loader)

        # Proper GPT forward (eval mode computes softcapped CE loss)
        schedule_cfg = ForwardScheduleConfig(
            mtp_weights=torch.ones(1, device=device),
            ws_short=w0.ws_short if hasattr(w0, 'ws_short') else 1024,
            ws_long=w0.ws_long if hasattr(w0, 'ws_long') else max_seq_len,
            train_max_seq_len=max_seq_len
        )
        gpt_loss = teacher_gpt(inputs, targets, cum_seqlens, bigram_inputs, schedule_cfg)
        loss_gpt += gpt_loss.item()

        # Wrapper forward (zeroed bigrams)
        wrapper_loss = teacher_wrapper.get_loss(inputs, targets, cum_seqlens, max_seq_len)
        loss_wrapper += wrapper_loss.item()

loss_gpt /= val_steps
loss_wrapper /= val_steps
print(f"  GPT proper val_loss:   {loss_gpt:.4f}")
print(f"  Wrapper val_loss:      {loss_wrapper:.4f}")
print(f"  Delta (wrapper worse): {loss_wrapper - loss_gpt:.4f}")

# ============================================================
# EXPERIMENT B: Logit distribution analysis
# ============================================================
print("\n" + "=" * 60)
print("EXPERIMENT B: Logit distribution analysis")
print("=" * 60)

# Fresh student for comparison
student_fresh = GPT(
    vocab_size=50257, num_layers=11, num_heads=6, head_dim=128, model_dim=768,
    max_seq_len=max_seq_len, kd_mode=True
).cuda()
for m in student_fresh.modules():
    if isinstance(m, (torch.nn.Embedding, torch.nn.Linear)):
        m.weight.data = m.weight.data.bfloat16()
student_fresh.attn_gate_bank.data = student_fresh.attn_gate_bank.data.bfloat16()
student_fresh.ve_gate_bank.data = student_fresh.ve_gate_bank.data.bfloat16()
student_fresh.attn_bank.data = student_fresh.attn_bank.data.bfloat16()
student_fresh.mlp_bank.data = student_fresh.mlp_bank.data.bfloat16()
student_fresh.eval()

# Get one batch
val_loader2 = distributed_data_generator(
    args.val_files, args.val_batch_size, -1,
    grad_accum_steps=grad_accum_steps, align_to_bos=False
)
inputs, targets, cum_seqlens, bigram_inputs, _ = next(val_loader2)

with torch.no_grad():
    # Teacher logits via wrapper (raw, no softcap)
    teacher_logits = teacher_wrapper.forward(inputs, cum_seqlens, max_seq_len)

    # Student logits (fresh, kd_mode returns (loss, logits))
    schedule_cfg = ForwardScheduleConfig(
        mtp_weights=torch.ones(1, device=device),
        ws_short=1024,
        ws_long=max_seq_len,
        train_max_seq_len=max_seq_len
    )
    _, student_logits = student_fresh(inputs, targets, cum_seqlens, bigram_inputs, schedule_cfg)

print(f"\n  Teacher logits: shape={teacher_logits.shape} dtype={teacher_logits.dtype}")
print(f"    absmax={teacher_logits.abs().max().item():.2f} mean={teacher_logits.mean().item():.4f} std={teacher_logits.std().item():.4f}")
print(f"  Student logits: shape={student_logits.shape} dtype={student_logits.dtype}")
print(f"    absmax={student_logits.abs().max().item():.2f} mean={student_logits.mean().item():.4f} std={student_logits.std().item():.4f}")

# Entropy and top-1 probability at different temperatures
for T in [1.0, 2.0, 5.0, 10.0]:
    t_probs = F.softmax(teacher_logits.float().view(-1, teacher_logits.size(-1)) / T, dim=-1)
    s_probs = F.softmax(student_logits.float().view(-1, student_logits.size(-1)) / T, dim=-1)

    t_entropy = -(t_probs * t_probs.clamp(min=1e-10).log()).sum(-1).mean().item() / np.log(2)
    s_entropy = -(s_probs * s_probs.clamp(min=1e-10).log()).sum(-1).mean().item() / np.log(2)

    t_top1 = t_probs.max(-1).values.mean().item()
    s_top1 = s_probs.max(-1).values.mean().item()

    print(f"\n  T={T:.1f}:")
    print(f"    Teacher: entropy={t_entropy:.2f} bits, top1_prob={t_top1:.4f}")
    print(f"    Student: entropy={s_entropy:.2f} bits, top1_prob={s_top1:.4f}")

del student_fresh
torch.cuda.empty_cache()

# ============================================================
# EXPERIMENT C: KD loss sanity checks
# ============================================================
print("\n" + "=" * 60)
print("EXPERIMENT C: KD loss sanity checks")
print("=" * 60)

with torch.no_grad():
    t_flat = teacher_logits.view(-1, teacher_logits.size(-1)).float()
    num_tokens = t_flat.size(0)

    # KL(teacher, teacher) should be 0
    t_log = F.log_softmax(t_flat, dim=-1)
    kl_self = F.kl_div(t_log, t_log, log_target=True, reduction='sum').item()
    kl_self_bm = F.kl_div(t_log, t_log, log_target=True, reduction='batchmean').item()
    print(f"\n  KL(teacher, teacher):")
    print(f"    sum reduction:       {kl_self:.6f}")
    print(f"    batchmean reduction: {kl_self_bm:.6f}")

    # KL(random, teacher) at various temperatures
    random_logits = torch.randn_like(t_flat) * 0.14  # ~student init scale
    for T in [1.0, 2.0, 5.0, 10.0]:
        r_log = F.log_softmax(random_logits / T, dim=-1)
        t_log_T = F.log_softmax(t_flat / T, dim=-1)

        kl_sum = F.kl_div(r_log, t_log_T, log_target=True, reduction='sum').item() * (T ** 2)
        kl_bm = F.kl_div(r_log, t_log_T, log_target=True, reduction='batchmean').item() * (T ** 2)

        print(f"\n  KL(random, teacher) at T={T:.1f}:")
        print(f"    sum reduction:       {kl_sum:.2f}")
        print(f"    batchmean reduction: {kl_bm:.4f}")
        print(f"    ratio (sum/batchmean): {kl_sum/kl_bm:.1f} (num_tokens={num_tokens})")

# ============================================================
# EXPERIMENT D: Gradient direction (cosine similarity)
# ============================================================
print("\n" + "=" * 60)
print("EXPERIMENT D: Gradient direction — CE vs KD")
print("=" * 60)

# Fresh student with grad
student_d = GPT(
    vocab_size=50257, num_layers=11, num_heads=6, head_dim=128, model_dim=768,
    max_seq_len=max_seq_len, kd_mode=True
).cuda()
for m in student_d.modules():
    if isinstance(m, (torch.nn.Embedding, torch.nn.Linear)):
        m.weight.data = m.weight.data.bfloat16()
student_d.attn_gate_bank.data = student_d.attn_gate_bank.data.bfloat16()
student_d.ve_gate_bank.data = student_d.ve_gate_bank.data.bfloat16()
student_d.attn_bank.data = student_d.attn_bank.data.bfloat16()
student_d.mlp_bank.data = student_d.mlp_bank.data.bfloat16()
student_d.train()

schedule_cfg = ForwardScheduleConfig(
    mtp_weights=torch.ones(1, device=device),
    ws_short=1024,
    ws_long=max_seq_len,
    train_max_seq_len=max_seq_len
)

# Gradient from hard CE loss
student_d.zero_grad()
hard_loss, s_logits = student_d(inputs, targets, cum_seqlens, bigram_inputs, schedule_cfg)
hard_loss.backward()
g_hard = torch.cat([p.grad.flatten().float() for p in student_d.parameters() if p.grad is not None])

# Get teacher logits once
with torch.no_grad():
    t_logits = teacher_wrapper.forward(inputs, cum_seqlens, max_seq_len)

for T in [1.0, 2.0, 5.0]:
    student_d.zero_grad()
    _, s_logits3 = student_d(inputs, targets, cum_seqlens, bigram_inputs, schedule_cfg)

    s_flat = s_logits3.view(-1, s_logits3.size(-1)) / T
    t_flat_d = t_logits.view(-1, t_logits.size(-1)).float() / T

    s_log = F.log_softmax(s_flat, dim=-1)
    t_log_d = F.log_softmax(t_flat_d, dim=-1)

    kd_loss = F.kl_div(s_log, t_log_d.detach(), log_target=True, reduction='batchmean') * (T ** 2)
    kd_loss.backward()
    g_soft = torch.cat([p.grad.flatten().float() for p in student_d.parameters() if p.grad is not None])

    cos_sim = F.cosine_similarity(g_hard.unsqueeze(0), g_soft.unsqueeze(0)).item()
    g_hard_norm = g_hard.norm().item()
    g_soft_norm = g_soft.norm().item()
    print(f"\n  T={T:.1f}: cosine_sim(g_hard, g_soft) = {cos_sim:.4f}")
    print(f"    |g_hard| = {g_hard_norm:.4f}, |g_soft| = {g_soft_norm:.4f}, ratio = {g_soft_norm/g_hard_norm:.4f}")

print("\n" + "=" * 60)
print("DIAGNOSTICS COMPLETE")
print("=" * 60)

dist.destroy_process_group()
