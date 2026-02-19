# Knowledge Distillation Debug Report

## Executive Summary

KD training in our modded-nanogpt-kd fork has been consistently underperforming the CE-only baseline. After extensive debugging, diagnostics, and comparison with the upstream working implementation, we identified **5 critical differences** and fixed them. The latest run shows improvement but KD still slightly trails the CE baseline.

## Project Context

### Repository Structure
- **Working repo**: `modded-nanogpt-kd/` (KD fork)
- **Upstream reference**: `modded-nanogpt/` (has working KD implementation)
- **Server**: `root@86.38.238.15` (H100 80GB), SSH key: `~/.ssh/prime_intellect_codex_ed25519`
- **Branch**: `kd-self-distill`
- **Teacher checkpoint**: `checkpoints/ce-baseline/state_step001530.pt` (val_loss 3.28)

### What We're Trying to Do
Train a student model with Knowledge Distillation using a previously-trained teacher. The teacher and student are the **same architecture** (self-distillation). The goal is for KD to help the student converge faster and/or to a better final loss than CE-only training.

### The Model
- GPT variant with 11 layers, 6 heads, head_dim=128, model_dim=768
- ~124M parameters
- Uses FP8 fused lm_head + softcap + cross-entropy kernel (FusedSoftcappedCrossEntropy)
- Training uses torch.compile with fullgraph=True
- Multi-stage training: W0 (steps 0-509), W1 (510-1019), W2 (1020-1530) with increasing batch sizes and window sizes
- Bigram hash embeddings contribute to every layer
- Value embeddings, attention gates, skip connections, smear gate, backout mechanism

### Key Files
- `train_gpt.py`: Main training script with GPT class, training loop, KD integration
- `teacher_model.py`: Contains `apply_kd_dynamic_norm` (and legacy TeacherGPT, StudentTeacherWrapper — now unused)
- `triton_kernels.py`: FusedSoftcappedCrossEntropy (our version: 7-arg fused matmul; upstream: 3-arg pre-computed logits)

---

## CE Baseline Results

Run with: `SAVE_CHECKPOINT=1 CKPT_DIR=checkpoints/ce-baseline torchrun --standalone --nproc_per_node=1 train_gpt.py`

| Step | Val Loss | Train Time | ms/step |
|------|----------|------------|---------|
| 0    | 10.83    | 0s         | -       |
| 250  | 4.50     | 59s        | 235     |
| 500  | 4.30     | 118s       | 236     |
| 750  | 3.86     | 227s       | 303     |
| 1000 | 3.56     | 337s       | 337     |
| 1250 | 3.39     | 498s       | 398     |
| 1500 | 3.29     | 658s       | 439     |
| 1530 | 3.28     | 677s       | 443     |

---

## KD Failure Timeline

### Attempt 1: Original TeacherGPT (separate class, soft-only loss)
- Teacher trained separately (val_loss ~4.11)
- Multiple bugs found and fixed:
  - `_orig_mod.` prefix from torch.compile not stripped → loaded zero weights
  - Used softcapped logits for KD (should be raw)
  - bf16 precision issues
  - `reduction='batchmean'` → changed to `'sum'` (this was wrong in retrospect)
- **Result**: Soft-only KD worked (val_loss 10.83 → 4.83 at step 750) but OOM at step 992

### Attempt 2: Combined hard+soft with dynamic norm
- Added `apply_kd_dynamic_norm` to scale soft_loss to match hard_loss
- KD auto-disables when val_loss < teacher_val_loss + 0.05
- User ran with wrong env var names first (ALPHA_SOFT instead of KD_ALPHA_SOFT)
- **Result**: Not properly tested due to env var mistake

### Attempt 3: Self-distillation with CE baseline as teacher
- Teacher checkpoint: `checkpoints/ce-baseline/state_step001530.pt` (val_loss 3.28)
- Used `StudentTeacherWrapper` to wrap GPT for teacher interface
- **Result**: Stuck at val_loss ~4.4, consistently worse than CE baseline

### Attempt 4: Fixed implementation (current)
- Rewrote to match upstream pattern
- **Result**: Closer but still slightly behind CE baseline

---

## Root Cause Analysis

### The 5 Critical Differences Between Our Fork and Upstream

#### 1. Teacher Model Architecture (FIXED)

**Before (broken):**
- Separate `TeacherGPT` class in `teacher_model.py` with simplified architecture
- OR `StudentTeacherWrapper` that zeros bigram embeddings and uses hardcoded window sizes
- Teacher val_loss: 3.92 (0.64 worse than actual model)

**After (fixed):**
- Same `GPT` class for teacher and student (like upstream)
- Teacher receives `bigram_inputs` and `schedule_cfg` from training loop
- Teacher val_loss: 3.49 (only 0.21 gap, likely from W0 window sizes)

**Upstream code:**
```python
teacher_model = GPT(vocab_size=50257, ...).cuda()
teacher_model.load_state_dict(state, strict=False)
teacher_model.eval()
teacher_logits = teacher_model(inputs, None, cum_seqlens, bigram_inputs,
                                forward_args, return_logits=True, return_loss=False)
```

#### 2. FusedSoftcappedCrossEntropy Divergence (PARTIALLY FIXED)

**Our fork's kernel** (7 args): `FusedSoftcappedCrossEntropy.apply(x, target, mtp_weights, lm_head.weight, x_s, w_s, grad_s)` — fuses FP8 lm_head matmul + softcap + CE. Never materializes logits.

**Upstream's kernel** (3 args): `FusedSoftcappedCrossEntropy.apply(logits, target, mtp_weights)` — takes pre-computed logits.

**Impact**: Our fork had to do TWO `lm_head(x)` calls in KD mode — one inside the fused kernel (invisible), one for KD logits. This caused:
- 650ms/step (vs 230ms baseline) — 2.8x slower
- Extra memory usage

**Current fix**: Dual-path approach:
- `return_logits=False` → uses fused FP8 kernel (fast, for pure CE)
- `return_logits=True` → uses unfused `lm_head(x)` + manual softcap + CE (for KD, single lm_head call)

**Remaining concern**: The unfused CE path may have different gradient characteristics than the fused kernel. The fused kernel does FP8 matmul in the backward pass; the unfused path uses regular bfloat16 matmul.

#### 3. KL Divergence Reduction (FIXED)

**Before**: `reduction='sum'` — raw sum over all tokens. With 16k tokens, loss ~160k.
**After**: `reduction='batchmean'` — per-token mean. Loss ~10.

**Impact**: With `sum` reduction + dynamic norm, the gradient was being rescaled every step to match hard_loss magnitude. But the dynamic norm ratio (hard/soft) was ~310k/160k ≈ 2x, which is stable. With `batchmean`, dynamic norm ratio is ~310k/10 ≈ 31k, which means soft_loss gets amplified 31k× — effectively making KD the dominant gradient signal.

This is actually what upstream does too (batchmean + dynamic norm). The question is whether this is correct.

#### 4. Alpha/Loss Combination (FIXED)

**Before**: `loss = alpha_hard * hard + alpha_soft * soft` with `alpha_hard=1.0, alpha_soft=1.0`
**After**: `loss = alpha * hard + (1-alpha) * soft` with `alpha=0.45`

Upstream also uses `lr_boost_mult=1.4` on the soft loss: `alpha * hard + (1-alpha) * lr_boost * soft`. We haven't added this yet.

#### 5. Temperature (FIXED)

**Before**: T=2.0 (user choice)
**After**: T=1.0 (upstream default)

Diagnostic showed at T=1: teacher entropy 1.27 bits, top-1 prob 76.6%. The distribution is very peaky — almost one-hot. This means at T=1, KD is essentially doing label smoothing toward the teacher's hard predictions.

---

## Diagnostic Results

### Experiment A: Teacher Quality
| Method | Val Loss | Gap from CE baseline |
|--------|----------|---------------------|
| StudentTeacherWrapper (zeroed bigrams) | 3.92 | +0.64 |
| Same GPT class (proper forward) | 3.49 | +0.21 |
| CE baseline actual | 3.28 | 0 |

### Experiment B: Logit Distribution
| | Teacher | Student (fresh) |
|---|---------|----------------|
| Absmax | 65.5 | 4.1 |
| Mean | -13.6 | 0.0 |
| Std | 5.2 | 0.14 |

**Entropy (bits) at different temperatures:**
| T | Teacher | Student | Max possible |
|---|---------|---------|-------------|
| 1.0 | 1.27 | 15.60 | 15.62 |
| 2.0 | 5.83 | 15.61 | 15.62 |
| 5.0 | 14.93 | 15.62 | 15.62 |
| 10.0 | 15.48 | 15.62 | 15.62 |

**Teacher top-1 probability:** 76.6% at T=1, 45.2% at T=2, 1.1% at T=5

### Experiment C: KL Loss Values
- KL(teacher, teacher) = 0.0 ✓
- KL(random, teacher) with batchmean:
  - T=1: 10.0, T=2: 30.0, T=5: 14.3, T=10: 10.8

---

## Current KD Run Results (latest fix)

Run with: `KD_ALPHA_SOFT=1.0 KD_ALPHA=0.45 KD_TEMPERATURE=1.0 KD_DYNAMIC_NORM=1`

| Step | KD Val Loss | CE Baseline | KD Better? |
|------|-------------|-------------|------------|
| 0    | 10.83       | 10.83       | Same       |
| 250  | 4.31        | 4.50        | **Yes (+0.19)** |
| 500  | 4.50        | 4.30        | No (-0.20) |
| 750  | 3.92        | 3.86        | No (-0.06) |
| (running...) | | | |

### Observations
1. KD helps at step 250 (faster early convergence)
2. KD hurts at step 500 (regression)
3. By step 750, gap narrows to 0.06
4. Soft loss decreasing: 10→2→0.5 (student converging toward teacher)
5. Step time: ~768ms (vs 235ms baseline) — 3.3x slower due to unfused CE path

---

## Open Questions and Hypotheses

### Q1: Why does val_loss regress at step 500?
Step 500 is right after the W0→W1 transition (step 509). Batch size doubles, window sizes change. The KD loss may be interfering with the transition.

### Q2: Is dynamic norm the right approach?
Dynamic norm makes soft_loss gradient ≈ hard_loss gradient in magnitude. But with alpha=0.45, the effective ratio is 0.45 * hard + 0.55 * (hard_magnitude_soft). If the soft gradient direction is even slightly wrong, it dominates.

### Q3: Should we use higher temperature?
At T=1, the teacher distribution is 76.6% one-hot. The "dark knowledge" (soft labels) barely exists. Upstream uses T=1 too, but their teacher might have different logit scales (they compute logits before the FP8 fused kernel, so precision differs).

### Q4: Is the unfused CE path causing gradient differences?
Our fused kernel does FP8 matmul → softcap → CE in one pass. The unfused path does bfloat16 matmul → softcap → CE. The backward gradients may differ due to precision.

### Q5: Does upstream's KD actually work better than CE baseline?
We haven't verified this. It's possible that KD with same-architecture self-distillation doesn't help even in the upstream repo.

### Q6: Should we add lr_boost_mult?
Upstream uses `lr_boost_mult=1.4` on the soft loss. We haven't added this. It would increase the KD signal.

---

## What's Been Tried

| Approach | Result |
|----------|--------|
| TeacherGPT (separate class) | Degraded teacher, stuck at 4.4 |
| StudentTeacherWrapper (zeroed bigrams) | 3.92 teacher val_loss, stuck at 4.4 |
| reduction='sum' | Extreme gradient magnitudes |
| reduction='batchmean' | Correct scale |
| T=2.0 | Over-smoothed teacher |
| T=1.0 | Very peaky teacher, minimal dark knowledge |
| alpha_hard=1.0, alpha_soft=1.0 | Equal weighting |
| alpha=0.45 (upstream) | Better, but still slightly worse than CE |
| Same GPT class for teacher | Fixed 0.64 loss degradation |
| return_logits dual-path | Single lm_head call, 768ms/step |

---

## Suggested Next Steps

1. **Wait for current run to finish** — compare final val_loss with CE baseline
2. **Test with lr_boost_mult=1.4** — matches upstream fully
3. **Test T=5** — more dark knowledge transfer
4. **Test soft-only (alpha=0)** — isolate whether soft loss alone converges
5. **Verify upstream KD works** — run upstream code with same teacher/student to establish ground truth
6. **Profile gradient differences** — compare fused vs unfused CE gradients
7. **Consider porting upstream's 3-arg FusedSoftcappedCrossEntropy** — unifies the code paths

---

## Code Reference

### GPT.forward() — the dual-path (train_gpt.py ~line 1359)
```python
if return_logits:
    # KD path: single lm_head call, unfused CE
    logits = self.lm_head(x)
    if return_loss:
        logits_sc = (23 * torch.sigmoid((logits + 5) / 7.5)).float()
        loss = F.cross_entropy(logits_sc.view(-1, ...), target_seq, reduction="sum")
else:
    # Pure CE: fused FP8 kernel (fast)
    if self.training:
        losses = FusedSoftcappedCrossEntropy.apply(x.view(-1, ...), target_seq, mtp_weights,
                    self.lm_head.weight, self.lm_head.x_s, self.lm_head.w_s, self.lm_head.grad_s)
        loss = losses.sum()
```

### Training loop KD section (train_gpt.py ~line 2204)
```python
if kd_enabled:
    student_logits, hard_loss = model(inputs, targets, cum_seqlens, bigram_inputs,
                                       forward_args, return_logits=True, return_loss=True)
    with torch.no_grad():
        teacher_logits = teacher_model(inputs, None, cum_seqlens, bigram_inputs,
                                       forward_args, return_logits=True, return_loss=False)
    T = args.kd_temperature
    s_log = F.log_softmax(student_logits.view(-1, V) / T, dim=-1)
    t_log = F.log_softmax(teacher_logits.float().view(-1, V) / T, dim=-1)
    soft_loss = F.kl_div(s_log, t_log.detach(), log_target=True, reduction='batchmean') * (T ** 2)
    if args.kd_dynamic_norm:
        soft_loss = apply_kd_dynamic_norm(hard_loss, soft_loss)
    loss = (alpha * hard_loss + (1.0 - alpha) * soft_loss) * grad_scale
```

### Upstream differences we haven't ported yet
- `lr_boost_mult=1.4` on soft loss
- `KD_SOFT_LOSS_TOKEN_CHUNK=4096` (chunked KL for memory)
- `KD_COMPILE_TEACHER=True` (compile teacher for speed)
- `KD_W0_ONLY=True` (disable KD after W0)
- LR post-stop schedule reduction
- Top-k KD loss filtering
