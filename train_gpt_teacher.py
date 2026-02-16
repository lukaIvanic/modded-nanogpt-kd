"""
train_gpt_teacher.py — Standalone teacher training for Knowledge Distillation.

Trains a small transformer (default: 3 layers, 6 heads, 128 head_dim, 768 d_model)
using pure cross-entropy loss. Saves checkpoint for later use in KD student training.

Based on modded-nanogpt by Keller Jordan et al.
Stripped of: U-net skips, value embeddings, bigram embeddings, paired heads,
sparse attention gates, smear/skip gates, backout, multi-token prediction.

Usage:
    torchrun --standalone --nproc_per_node=1 train_gpt_teacher.py

Environment variables:
    NUM_LAYERS       (default: 3)
    NUM_HEADS        (default: 6)
    HEAD_DIM         (default: 128)
    NUM_ITERATIONS   (default: 250)
    BATCH_SIZE       (default: 8*2048*8 = 131072)
    WARMUP_FRAC      (default: 0.05)
    VAL_LOSS_EVERY   (default: 25)
    SAVE_CHECKPOINT  (default: 1)
    CHECKPOINT_DIR   (default: checkpoints/teacher)
    DATA_PATH        (default: .)
"""

import os
import sys
import copy
import glob
import math
import threading
import time
import uuid
import gc
from dataclasses import dataclass
from pathlib import Path

# Read source code for logging
with open(sys.argv[0], 'r') as f:
    code = f.read()
with open(os.path.join(os.path.dirname(sys.argv[0]), 'triton_kernels.py'), 'r') as f:
    code += f"\n\n{'-'*40}\n# triton_kernels.py\n{'-'*40}\n\n"
    code += f.read()

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
import torch
import triton
import numpy as np

torch.empty(
    1, device=f"cuda:{os.environ['LOCAL_RANK']}", requires_grad=True
).backward()  # prevents a bug on some systems
import torch._dynamo as dynamo
import torch.distributed as dist
import torch.nn.functional as F

from kernels import get_kernel
from torch import Tensor, nn
from triton_kernels import XXT, ba_plus_cAA, FusedLinearReLUSquareFunction, FusedSoftcappedCrossEntropy

dynamo.config.recompile_limit = 64

# -----------------------------------------------------------------------------
# Distributed training setup
rank = int(os.environ["RANK"])
world_size = int(os.environ["WORLD_SIZE"])
assert 8 % world_size == 0, "world_size must be a divisor of 8"
grad_accum_steps = 8 // world_size
grad_scale = 1 / grad_accum_steps
assert torch.cuda.is_available()
device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
torch.cuda.set_device(device)
dist.init_process_group(backend="cuda:nccl,cpu:gloo", device_id=device)
dist.barrier()
master_process = (rank == 0)

# -----------------------------------------------------------------------------
# Polar Express orthogonalization for NorMuon optimizer

polar_express_coeffs = [
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323)
]

@torch.compile(dynamic=False, fullgraph=True)
def polar_express(G: torch.Tensor, split_baddbmm: bool = False):
    """Polar Express Sign Method: https://arxiv.org/pdf/2505.16932"""
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * (1 + 2e-2) + 1e-6)
    X = X.contiguous()
    A = torch.empty((*X.shape[:-1], X.size(-2)), device=X.device, dtype=X.dtype)
    B = torch.empty_like(A)
    C = torch.empty_like(X)
    if split_baddbmm:
        BX_matmul = torch.bmm if X.ndim > 2 else torch.mm
    else:
        aX_plus_BX = torch.baddbmm if X.ndim > 2 else torch.addmm
    for a, b, c in polar_express_coeffs:
        XXT(X, out=A)
        ba_plus_cAA(A, alpha=c, beta=b, out=B)
        if split_baddbmm:
            BX_matmul(B, X, out=C)
            C.add_(X, alpha=a)
        else:
            aX_plus_BX(X, B, X, beta=a, out=C)
        X, C = C, X
    if G.size(-2) > G.size(-1):
        X = X.mT
    return X

# -----------------------------------------------------------------------------
# NorMuonAndAdam optimizer (simplified: no sparse comms)

@dataclass
class ParamConfig:
    label: str
    optim: str          # "adam" or "normuon"
    comms: str          # "none", "replicated", "sharded"
    adam_betas: tuple[float, float] | None
    lr_mul: float
    wd_mul: float
    lr: float
    initial_lr: float
    weight_decay: float
    eps: float | None = None
    reshape: tuple | None = None
    chunk_size: int | None = None
    momentum: float | None = None
    beta2: float | None = None
    per_matrix_lr_mul: list[float] | None = None


class NorMuonAndAdam:
    """Combined Muon (for projection matrices) + Adam (for embeddings/scalars) optimizer."""

    def __init__(self, named_params, param_table: dict, scatter_order: list, work_order: list,
                 adam_defaults: dict, normuon_defaults: dict):
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        self.adam_defaults = adam_defaults
        self.normuon_defaults = normuon_defaults
        self.param_table = param_table
        self.scatter_order = scatter_order
        self.work_order = work_order

        self.param_cfgs: dict[nn.Parameter, ParamConfig] = {}
        self.param_states: dict[nn.Parameter, dict] = {}
        self._param_by_label: dict[str, nn.Parameter] = {}
        for name, param in named_params:
            label = getattr(param, "label", None)
            assert label is not None and label in param_table
            assert label not in self._param_by_label
            self._param_by_label[label] = param
            self._build_param_cfg(param, label)

        present = set(self._param_by_label.keys())
        assert set(scatter_order) == present and set(work_order) == present

        if self.world_size == 1:
            for p_cfg in self.param_cfgs.values():
                p_cfg.comms = "none"

        self._init_state()
        self._step_size_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._eff_wd_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._eff_lr_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._reduce_futures: dict[nn.Parameter, tuple] = {}

        self.split_embed = False
        self._lm_head_param = self._param_by_label.get("lm_head")
        self._embed_param = self._param_by_label.get("embed")

    def _build_param_cfg(self, param: nn.Parameter, label: str):
        table_entry = self.param_table[label]
        optim = table_entry["optim"]
        comms = table_entry["comms"]
        adam_betas = table_entry.get("adam_betas")
        lr_mul = table_entry.get("lr_mul", 1.0)
        wd_mul = table_entry.get("wd_mul", 1.0)

        if optim == "adam":
            chunk_size = param.shape[0] // self.world_size if comms.startswith("sharded") else None
            p_cfg = ParamConfig(
                label=label, optim=optim, comms=comms,
                adam_betas=tuple(adam_betas) if adam_betas else None,
                lr_mul=lr_mul, wd_mul=wd_mul,
                lr=self.adam_defaults["lr"], initial_lr=self.adam_defaults["lr"],
                weight_decay=self.adam_defaults["weight_decay"],
                eps=self.adam_defaults["eps"], chunk_size=chunk_size,
            )
        elif optim == "normuon":
            reshape = getattr(param, "reshape", None)
            if reshape is None:
                raise ValueError(f"NorMuon param {label} must have .reshape attribute")
            if reshape[0] % self.world_size != 0:
                raise ValueError(f"reshape[0]={reshape[0]} must be divisible by world_size")
            chunk_size = reshape[0] // self.world_size
            chunk_shape = (chunk_size, *reshape[1:])
            shape_mult = max(1.0, chunk_shape[-2] / chunk_shape[-1]) ** 0.5 if len(chunk_shape) >= 2 else 1.0
            lr_mul = shape_mult * lr_mul

            per_matrix_lr_mul = None
            if label == "mlp":
                r = dist.get_rank() if dist.is_initialized() else 0
                start_idx = r * chunk_size
                per_matrix_lr_mul = [2.0 if (start_idx + i) % 2 == 1 else 1.0 for i in range(chunk_size)]

            p_cfg = ParamConfig(
                label=label, optim=optim, comms=comms,
                adam_betas=tuple(adam_betas) if adam_betas else None,
                lr_mul=lr_mul, wd_mul=wd_mul,
                lr=self.normuon_defaults["lr"], initial_lr=self.normuon_defaults["lr"],
                weight_decay=self.normuon_defaults["weight_decay"],
                reshape=reshape, chunk_size=chunk_size,
                momentum=self.normuon_defaults["momentum"],
                beta2=self.normuon_defaults["beta2"],
                per_matrix_lr_mul=per_matrix_lr_mul,
            )
        else:
            raise ValueError(f"Unknown optim type: {optim}")
        self.param_cfgs[param] = p_cfg

    def _init_state(self):
        for param, p_cfg in self.param_cfgs.items():
            if p_cfg.optim == "adam":
                chunk = param[:p_cfg.chunk_size] if p_cfg.comms.startswith("sharded") else param
                exp_avg = torch.zeros_like(chunk, dtype=torch.float32, device=param.device)
                self.param_states[param] = dict(step=0, exp_avg=exp_avg, exp_avg_sq=torch.zeros_like(exp_avg))
            elif p_cfg.optim == "normuon":
                chunk_shape = (p_cfg.chunk_size, *p_cfg.reshape[1:])
                momentum_buffer = torch.zeros(chunk_shape, dtype=torch.float32, device=param.device)
                if chunk_shape[-2] >= chunk_shape[-1]:
                    second_mom_shape = (*chunk_shape[:-1], 1)
                else:
                    second_mom_shape = (*chunk_shape[:-2], 1, chunk_shape[-1])
                second_momentum_buffer = torch.zeros(second_mom_shape, dtype=torch.float32, device=param.device)
                mantissa = torch.zeros(chunk_shape, dtype=torch.uint16, device=param.device)
                self.param_states[param] = dict(
                    momentum_buffer=momentum_buffer,
                    second_momentum_buffer=second_momentum_buffer,
                    mantissa=mantissa,
                )

    def _launch_reduce(self, param: nn.Parameter, grad: Tensor):
        p_cfg = self.param_cfgs[param]
        if p_cfg.comms == "none":
            if p_cfg.optim == "normuon":
                grad = grad.view(p_cfg.reshape)
            self._reduce_futures[param] = (None, grad)
        elif p_cfg.comms == "replicated":
            future = dist.all_reduce(grad, op=dist.ReduceOp.AVG, async_op=True).get_future()
            self._reduce_futures[param] = (future, grad)
        elif p_cfg.comms == "sharded":
            if p_cfg.optim == "normuon":
                grad_reshaped = grad.view(p_cfg.reshape)
                grad_chunk = torch.empty(
                    (p_cfg.chunk_size, *grad_reshaped.shape[1:]), dtype=grad.dtype, device=grad.device)
                future = dist.reduce_scatter_tensor(
                    grad_chunk, grad_reshaped.contiguous(), op=dist.ReduceOp.AVG, async_op=True).get_future()
                self._reduce_futures[param] = (future, grad_chunk)
            else:
                grad_chunk = torch.empty_like(grad[:p_cfg.chunk_size])
                future = dist.reduce_scatter_tensor(
                    grad_chunk, grad, op=dist.ReduceOp.AVG, async_op=True).get_future()
                self._reduce_futures[param] = (future, grad_chunk)

    def _launch_gather(self, param: nn.Parameter, p_slice: Tensor):
        p_cfg = self.param_cfgs[param]
        if p_cfg.optim == "normuon":
            full_param = param.data.view(p_cfg.reshape)
            assert full_param.is_contiguous()
            return dist.all_gather_into_tensor(full_param, p_slice.contiguous(), async_op=True).get_future()
        else:
            return dist.all_gather_into_tensor(param, p_slice.contiguous(), async_op=True).get_future()

    def reset(self):
        self.split_embed = False
        for param, p_cfg in self.param_cfgs.items():
            if p_cfg.optim == "normuon":
                p_state = self.param_states[param]
                p_state["momentum_buffer"].zero_()
                p_state["mantissa"].zero_()
                p_state["second_momentum_buffer"].zero_()

    def copy_lm_state_to_embed(self):
        lm_head = self._lm_head_param
        embed = self._embed_param
        lm_state = self.param_states[lm_head]
        embed_state = self.param_states[embed]
        lm_cfg = self.param_cfgs[lm_head]
        embed_cfg = self.param_cfgs[embed]
        embed_state['step'] = lm_state['step']
        if self.world_size > 1:
            r = dist.get_rank()
            embed_chunk_size = embed_cfg.chunk_size
            for key in ["exp_avg", "exp_avg_sq"]:
                lm_chunk = lm_state[key]
                full_lm = torch.empty(lm_head.shape[0], lm_head.shape[1], dtype=lm_chunk.dtype, device=lm_chunk.device)
                dist.all_gather_into_tensor(full_lm, lm_chunk.contiguous())
                embed_state[key].copy_(full_lm.T[r * embed_chunk_size:(r + 1) * embed_chunk_size])
        else:
            for key in ["exp_avg", "exp_avg_sq"]:
                embed_state[key].copy_(lm_state[key].T)
        self.split_embed = True

    def state_dict(self):
        return {
            "param_states": {id(p): s for p, s in self.param_states.items()},
            "param_cfgs": {id(p): s for p, s in self.param_cfgs.items()},
        }

    def load_state_dict(self, state_dict):
        id_to_param = {id(p): p for p in self.param_cfgs.keys()}
        for param_id, saved_p_state in state_dict["param_states"].items():
            if param_id in id_to_param:
                param = id_to_param[param_id]
                p_state = self.param_states[param]
                for k, v in saved_p_state.items():
                    if isinstance(v, torch.Tensor) and k in p_state:
                        p_state[k] = v.to(dtype=p_state[k].dtype, device=p_state[k].device)
                    else:
                        p_state[k] = v

    @torch.no_grad()
    def step(self, do_adam: bool = True):
        r = dist.get_rank() if dist.is_initialized() else 0
        lm_param, embed_param = self._lm_head_param, self._embed_param

        # Phase 1: Launch reduces
        for label in self.scatter_order:
            param = self._param_by_label[label]
            p_cfg = self.param_cfgs[param]
            if p_cfg.optim == "adam" and not do_adam:
                continue
            if param.grad is None:
                continue
            if label == "lm_head" and do_adam and not self.split_embed:
                if embed_param is not None and embed_param.grad is not None:
                    param.grad.add_(embed_param.grad.T)
            if label == "embed" and not self.split_embed:
                continue
            self._launch_reduce(param, param.grad)

        # Phase 2: Process updates
        gather_futures = []
        lm_head_gather_future = None
        for label in self.work_order:
            param = self._param_by_label[label]
            if param not in self._reduce_futures:
                continue
            p_cfg = self.param_cfgs[param]
            if p_cfg.optim == "adam" and not do_adam:
                continue
            future, grad_chunk = self._reduce_futures[param]
            if future is not None:
                future.wait()
            if p_cfg.optim == "adam":
                p_slice = self._adam_update(param, grad_chunk, p_cfg, r)
            else:
                p_slice = self._normuon_update(param, grad_chunk, p_cfg, r)
            if p_cfg.comms.startswith("sharded") and self.world_size > 1:
                gather_fut = self._launch_gather(param, p_slice)
                if label == "lm_head":
                    lm_head_gather_future = gather_fut
                else:
                    gather_futures.append(gather_fut)

        # Phase 3: Wait for gathers
        if lm_head_gather_future is not None:
            lm_head_gather_future.wait()
        if do_adam and not self.split_embed and embed_param is not None and lm_param is not None:
            embed_param.data.copy_(lm_param.data.T)
        for fut in gather_futures:
            fut.wait()
        self._reduce_futures.clear()

        for param, p_cfg in self.param_cfgs.items():
            if p_cfg.optim == "adam" and not do_adam:
                continue
            param.grad = None

    def _adam_update(self, param, grad_chunk, p_cfg, r):
        lr = p_cfg.lr * p_cfg.lr_mul
        if p_cfg.comms.startswith("sharded"):
            p_slice = param[r * p_cfg.chunk_size:(r + 1) * p_cfg.chunk_size]
        else:
            p_slice = param
        p_state = self.param_states[param]
        p_state["step"] += 1
        t = p_state["step"]
        bias1, bias2 = 1 - p_cfg.adam_betas[0] ** t, 1 - p_cfg.adam_betas[1] ** t
        self._step_size_t.fill_(lr * (bias2 ** 0.5 / bias1))
        self._eff_wd_t.fill_(lr * lr * p_cfg.weight_decay * p_cfg.wd_mul)
        NorMuonAndAdam._adam_update_step(
            p_slice, grad_chunk, p_state["exp_avg"], p_state["exp_avg_sq"],
            p_cfg.adam_betas[0], p_cfg.adam_betas[1], p_cfg.eps, self._step_size_t, self._eff_wd_t)
        return p_slice

    @staticmethod
    @torch.compile(dynamic=False, fullgraph=True)
    def _adam_update_step(p_slice, g_slice, exp_avg, exp_avg_sq, beta1, beta2, eps, step_size_t, eff_wd_t):
        exp_avg.mul_(beta1).add_(g_slice, alpha=1 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(g_slice, g_slice, value=1 - beta2)
        update = exp_avg.div(exp_avg_sq.sqrt().add_(eps)).mul_(step_size_t)
        mask = (update * p_slice) > 0
        update.addcmul_(p_slice, mask, value=eff_wd_t)
        p_slice.add_(other=update, alpha=-1.0)

    def _normuon_update(self, param, grad_chunk, p_cfg, r):
        p_state = self.param_states[param]
        grad_chunk = grad_chunk.float()
        momentum_buffer = p_state["momentum_buffer"]
        momentum_buffer.lerp_(grad_chunk, 1 - p_cfg.momentum)
        updated_grads = grad_chunk.lerp_(momentum_buffer, p_cfg.momentum)
        self._eff_lr_t.fill_(p_cfg.lr_mul * p_cfg.lr)
        self._eff_wd_t.fill_(p_cfg.wd_mul * p_cfg.weight_decay * p_cfg.lr)
        is_large_matrix = grad_chunk.shape[-2] > 1024
        v_chunk = polar_express(updated_grads, split_baddbmm=is_large_matrix)
        red_dim = -1 if grad_chunk.shape[-2] >= grad_chunk.shape[-1] else -2
        v_chunk = NorMuonAndAdam._apply_normuon_variance_reduction(
            v_chunk, p_state["second_momentum_buffer"], p_cfg.beta2, red_dim)
        param_view = param.data.view(p_cfg.reshape)
        p_slice = param_view[r * p_cfg.chunk_size:(r + 1) * p_cfg.chunk_size]
        if p_cfg.per_matrix_lr_mul is not None:
            for mat_idx in range(p_cfg.chunk_size):
                self._eff_lr_t.fill_(p_cfg.lr_mul * p_cfg.per_matrix_lr_mul[mat_idx] * p_cfg.lr)
                self._eff_wd_t.fill_(p_cfg.wd_mul * p_cfg.weight_decay * p_cfg.lr)
                NorMuonAndAdam._cautious_wd_and_update_inplace(
                    p_slice[mat_idx].view(torch.uint16), p_state["mantissa"][mat_idx], v_chunk[mat_idx],
                    self._eff_wd_t, self._eff_lr_t)
        else:
            NorMuonAndAdam._cautious_wd_and_update_inplace(
                p_slice.view(torch.uint16), p_state["mantissa"], v_chunk,
                self._eff_wd_t, self._eff_lr_t)
        return p_slice

    @staticmethod
    @torch.compile(dynamic=False, fullgraph=True)
    def _cautious_wd_and_update_inplace(p, mantissa, grad, wd_tensor, lr_tensor):
        assert p.dtype == mantissa.dtype == torch.uint16
        grad = grad.float()
        wd_factor = wd_tensor.to(torch.float32)
        lr_factor = lr_tensor.to(torch.float32)
        p_precise_raw = (p.to(torch.uint32) << 16) | mantissa.to(torch.uint32)
        p_precise = p_precise_raw.view(torch.float32)
        mask = (grad * p_precise) >= 0
        p_precise.copy_(p_precise - (p_precise * mask * wd_factor * lr_factor) - (grad * lr_factor))
        p.copy_((p_precise_raw >> 16).to(torch.uint16))
        mantissa.copy_(p_precise_raw.to(torch.uint16))

    @staticmethod
    @torch.compile(dynamic=False, fullgraph=True)
    def _apply_normuon_variance_reduction(v_chunk, second_momentum_buffer, beta2, red_dim):
        v_mean = v_chunk.float().square().mean(dim=red_dim, keepdim=True)
        red_dim_size = v_chunk.size(red_dim)
        v_norm_sq = v_mean.sum(dim=(-2, -1), keepdim=True).mul_(red_dim_size)
        v_norm = v_norm_sq.sqrt_()
        second_momentum_buffer.lerp_(v_mean.to(dtype=second_momentum_buffer.dtype), 1 - beta2)
        step_size = second_momentum_buffer.clamp_min(1e-10).rsqrt_()
        scaled_sq_sum = (v_mean * red_dim_size) * step_size.float().square()
        v_norm_new = scaled_sq_sum.sum(dim=(-2, -1), keepdim=True).sqrt_()
        final_scale = step_size * (v_norm / v_norm_new.clamp_min_(1e-10))
        return v_chunk.mul_(final_scale.type_as(v_chunk))


# -----------------------------------------------------------------------------
# Model components

def norm(x: Tensor):
    return F.rms_norm(x, (x.size(-1),))

def next_multiple_of_n(v: float | int, *, n: int):
    return next(x for x in range(n, int(v) + 1 + n, n) if x >= v)


class Yarn(nn.Module):
    """RoPE with YaRN frequency extension."""
    def __init__(self, head_dim, max_seq_len):
        super().__init__()
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.reset()

    def rotary(self, x_BTHD):
        assert self.factor1.size(0) >= x_BTHD.size(-3)
        factor1 = self.factor1[None, :x_BTHD.size(-3), None, :]
        factor2 = self.factor2[None, :x_BTHD.size(-3), None, :]
        x_flip = x_BTHD.view(*x_BTHD.shape[:-1], x_BTHD.shape[-1] // 2, 2).flip(-1).view(x_BTHD.shape)
        return factor1 * x_BTHD + factor2 * x_flip

    def reset(self):
        angular_freq = (1 / 1024) ** torch.linspace(0, 1, steps=self.head_dim // 4, dtype=torch.float32, device=device)
        angular_freq = angular_freq.repeat_interleave(2)
        angular_freq = torch.cat([angular_freq, angular_freq.new_zeros(self.head_dim // 2)])
        t = torch.arange(2 * self.max_seq_len, dtype=torch.float32, device=device)
        theta = torch.outer(t, angular_freq)
        self.factor1 = nn.Buffer(theta.cos().to(torch.bfloat16), persistent=False)
        self.factor2 = nn.Buffer(theta.sin().to(torch.bfloat16), persistent=False)
        self.factor2[..., 1::2] *= -1
        self.angular_freq = angular_freq
        self.attn_scale = 0.1


flash_attn_interface = get_kernel('varunneal/flash-attention-3').flash_attn_interface


class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, head_dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dim = dim

    def forward(self, x: Tensor, seqlens: Tensor, max_len: int, yarn: Yarn, sa_lambdas: Tensor, qkvo_w: Tensor):
        B, T = x.size(0), x.size(1)
        assert B == 1
        q, k, v = F.linear(x, sa_lambdas[0] * qkvo_w[:self.dim * 3].type_as(x)).view(B, T, 3 * self.num_heads, self.head_dim).chunk(3, dim=-2)
        q, k = norm(q), norm(k)
        q, k = yarn.rotary(q), yarn.rotary(k)
        y = flash_attn_interface.flash_attn_varlen_func(
            q[0], k[0], v[0],
            cu_seqlens_q=seqlens, cu_seqlens_k=seqlens,
            max_seqlen_q=max_len, max_seqlen_k=max_len,
            causal=True, softmax_scale=yarn.attn_scale,
            window_size=(-1, -1))  # full causal, no sliding window
        y = y.view(B, T, self.num_heads * self.head_dim)
        y = F.linear(y, sa_lambdas[1] * qkvo_w[self.dim * 3:].type_as(y))
        return y


class MLP(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor, c_fc: Tensor, c_proj: Tensor):
        return FusedLinearReLUSquareFunction.apply(x, c_fc, c_proj)


class Block(nn.Module):
    def __init__(self, dim: int, head_dim: int, num_heads: int):
        super().__init__()
        self.attn = CausalSelfAttention(dim, head_dim, num_heads)
        self.mlp = MLP()

    def forward(self, x, seqlens, max_len, yarn, sa_lambdas, qkvo_w, c_fc, c_proj):
        x = x + self.attn(norm(x), seqlens, max_len, yarn, sa_lambdas, qkvo_w)
        x = x + self.mlp(norm(x), c_fc, c_proj)
        return x


class GPT(nn.Module):
    def __init__(self, vocab_size: int, num_layers: int, num_heads: int, head_dim: int, model_dim: int, max_seq_len: int):
        super().__init__()
        self.num_layers = num_layers
        self.vocab_size = next_multiple_of_n(vocab_size, n=128)

        hdim = num_heads * head_dim
        mlp_hdim = 4 * model_dim

        # Attention weight bank: (num_layers+padding, 4*model_dim, hdim)
        # Padding ensures leading dim * 4 is divisible by world_size for sharding
        num_attn_with_padding = num_layers
        while (num_attn_with_padding * 4) % world_size != 0:
            num_attn_with_padding += 1
        self.attn_bank = nn.Parameter(torch.empty(num_attn_with_padding, 4 * model_dim, hdim))
        self.attn_bank.label = 'attn'
        self.attn_bank.reshape = (num_attn_with_padding * 4, hdim, hdim)

        # MLP weight bank: (num_layers+padding, 2, mlp_hdim, model_dim)
        num_mlp_with_padding = num_layers
        while (num_mlp_with_padding * 2) % world_size != 0:
            num_mlp_with_padding += 1
        self.mlp_bank = nn.Parameter(torch.empty(num_mlp_with_padding, 2, mlp_hdim, model_dim))
        self.mlp_bank.label = 'mlp'
        self.mlp_bank.reshape = (num_mlp_with_padding * 2, mlp_hdim, model_dim)

        # Init weights
        std = 0.5 * model_dim ** -0.5
        bound = (3 ** 0.5) * std
        with torch.no_grad():
            self.attn_bank.uniform_(-bound, bound)
            self.mlp_bank[:, 0, :, :].uniform_(-bound, bound)
            self.mlp_bank[:, 1, :, :].zero_()

        self.blocks = nn.ModuleList([
            Block(model_dim, head_dim, num_heads) for _ in range(num_layers)
        ])
        self.yarn = Yarn(head_dim, max_seq_len)

        use_fp8 = not os.environ.get("DISABLE_FP8", False)
        self.lm_head = CastedLinearT(model_dim, self.vocab_size, use_fp8=use_fp8, x_s=100/448, w_s=1.6/448, grad_s=grad_scale * 0.75/448)
        nn.init.normal_(self.lm_head.weight, mean=0, std=0.005)
        self.lm_head.weight.label = 'lm_head'

        self.embed = nn.Embedding(self.vocab_size, model_dim)
        self.embed.weight.label = 'embed'
        with torch.no_grad():
            self.embed.weight.copy_(self.lm_head.weight.T)

        # Residual scaling: resid_lambdas + sa_lambdas + x0_lambdas
        self.x0_lambdas = nn.Parameter(torch.zeros(num_layers))
        self.x0_lambdas.label = 'x0_lambdas'

        # Scalars: resid_lambdas (num_layers) + sa_lambdas (2*num_layers)
        num_scalars = 3 * num_layers
        pad = (-num_scalars) % max(world_size, 1)
        self.scalars = nn.Parameter(torch.cat([
            1.1 * torch.ones(num_layers),                              # resid_lambdas
            *[torch.tensor([0.5, 1.0]) for _ in range(num_layers)],    # sa_lambdas
            torch.zeros(pad),                                          # padding
        ]))
        self.scalars.label = 'scalars'

    def forward(self, input_seq: Tensor, target_seq: Tensor, cum_seqlens: Tensor):
        # Compute max_len for flash attention
        if self.training:
            max_len = train_max_seq_len
        else:
            max_len = hparams.val_batch_size // (grad_accum_steps * world_size)

        # Unpack scalars
        resid_lambdas = self.scalars[:self.num_layers]
        x0_lambdas = self.x0_lambdas
        sa_lambdas = self.scalars[self.num_layers:3 * self.num_layers].view(-1, 2)

        x = self.embed(input_seq)
        x = x0 = norm(x[None])

        attn_weights = self.attn_bank[:self.num_layers].unbind(0)
        mlp_fcs = self.mlp_bank[:self.num_layers, 0, :, :].unbind(0)
        mlp_projs = self.mlp_bank[:self.num_layers, 1, :, :].unbind(0)

        for i in range(self.num_layers):
            if i == 0:
                x = (resid_lambdas[0] + x0_lambdas[0]) * x
            else:
                x = resid_lambdas[i] * x + x0_lambdas[i] * x0
            x = self.blocks[i](x, cum_seqlens, max_len, self.yarn, sa_lambdas[i], attn_weights[i], mlp_fcs[i], mlp_projs[i])

        x = norm(x)
        # Softcapped cross-entropy
        if self.training:
            losses = FusedSoftcappedCrossEntropy.apply(
                x.view(-1, x.size(-1)), target_seq, torch.ones(1, device=x.device),
                self.lm_head.weight, self.lm_head.x_s, self.lm_head.w_s, self.lm_head.grad_s)
            loss = losses.sum()
        else:
            logits = self.lm_head(x)
            logits = 23 * torch.sigmoid((logits + 5) / 7.5)
            loss = F.cross_entropy(logits.float().view(-1, logits.size(-1)), target_seq, reduction="mean")
        return loss


# FP8 operators (needed by CastedLinearT)
@torch.library.custom_op("nanogpt_teacher::mm_t", mutates_args=())
def mm_t_op(x: Tensor, w: Tensor, x_s: float = 1.0, w_s: float = 1.0, grad_s: float = 1.0) -> tuple[Tensor, Tensor, Tensor]:
    x_f8 = (x * x_s).to(torch.float8_e4m3fn)
    w_f8 = (w * w_s).to(torch.float8_e4m3fn)
    out = torch._scaled_mm(x_f8, w_f8, out_dtype=torch.bfloat16, use_fast_accum=True, scale_a=torch.tensor(1/x_s), scale_b=torch.tensor(1/w_s))
    return out, x_f8, w_f8

@mm_t_op.register_fake
def mm_t_op_fake(x, w, x_s=1.0, w_s=1.0, grad_s=1.0):
    out = (x @ w).to(torch.bfloat16)
    return out, x.to(torch.float8_e4m3fn), w.to(torch.float8_e4m3fn)

@torch.library.custom_op("nanogpt_teacher::mm_t_backward", mutates_args=())
def mm_t_backward_op(grad: Tensor, x_f8: Tensor, w_f8: Tensor, x_s: float, w_s: float, grad_s: float) -> tuple[Tensor, Tensor]:
    g_f8 = (grad * grad_s).to(torch.float8_e4m3fn)
    grad_x = torch._scaled_mm(g_f8, w_f8.t(), out_dtype=torch.bfloat16, use_fast_accum=True, scale_a=torch.tensor(1/grad_s), scale_b=torch.tensor(1/w_s))
    grad_w = torch._scaled_mm(x_f8.t(), g_f8, out_dtype=torch.float32, use_fast_accum=False, scale_a=torch.tensor(1/x_s), scale_b=torch.tensor(1/grad_s))
    return grad_x, grad_w

@mm_t_backward_op.register_fake
def mm_t_backward_op_fake(grad, x_f8, w_f8, x_s, w_s, grad_s):
    return (grad @ w_f8.t().to(grad.dtype)).to(torch.bfloat16), (x_f8.t().to(grad.dtype) @ grad).to(torch.float32)

def backward_t(ctx, grad_output, *args):
    x_s, w_s, grad_s = ctx.x_s, ctx.w_s, ctx.grad_s
    grad_x, grad_w = torch.ops.nanogpt_teacher.mm_t_backward(grad_output, ctx.x_f8, ctx.w_f8, x_s, w_s, grad_s)
    return grad_x, grad_w, None, None, None

def setup_context_t(ctx, inputs, output):
    x, w, x_s, w_s, grad_s = inputs
    out, x_f8, w_f8 = output
    ctx.x_f8 = x_f8
    ctx.w_f8 = w_f8
    ctx.x_s = x_s
    ctx.w_s = w_s
    ctx.grad_s = grad_s

torch.library.register_autograd("nanogpt_teacher::mm_t", backward_t, setup_context=setup_context_t)


class CastedLinearT(nn.Module):
    """Linear with transposed weight storage, optional FP8."""
    def __init__(self, in_features, out_features, use_fp8=False, x_s=1.0, w_s=1.0, grad_s=1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_fp8 = use_fp8
        self.x_s = x_s
        self.w_s = w_s
        self.grad_s = grad_s
        self.weight = nn.Parameter(torch.empty(in_features, out_features, dtype=torch.bfloat16))
        with torch.no_grad():
            nn.init.zeros_(self.weight)

    def forward(self, x):
        if self.use_fp8 and self.training:
            _x = x.flatten(0, -2)
            out = torch.ops.nanogpt_teacher.mm_t(_x, self.weight, x_s=self.x_s, w_s=self.w_s, grad_s=self.grad_s)[0]
            return out.reshape(*x.shape[:-1], -1)
        else:
            return x @ self.weight.type_as(x)


# -----------------------------------------------------------------------------
# Data loading (simplified: no bigram hash)

BOS_ID = 50256

def _load_data_shard(file: Path):
    header = torch.from_file(str(file), False, 256, dtype=torch.int32)
    assert header[0] == 20240520, "magic number mismatch"
    assert header[1] == 1, "unsupported version"
    num_tokens = int(header[2])
    with file.open("rb", buffering=0) as f:
        tokens = torch.empty(num_tokens, dtype=torch.uint16, pin_memory=True)
        f.seek(256 * 4)
        nbytes = f.readinto(tokens.numpy())
        assert nbytes == 2 * num_tokens
    return tokens


class Shard:
    def __init__(self, tokens: Tensor, world_size: int = 1):
        self.tokens = tokens
        self.size = tokens.numel()
        self.world_size = world_size
        self.i = 0
        self.bos_idx = (tokens[:6_000_000] == BOS_ID).nonzero(as_tuple=True)[0].to(torch.int64).cpu().numpy()
        self._full_idx = None
        self._ready = threading.Event()
        self._loader_thread = threading.Thread(target=self._scan)
        self._loader_thread.start()

    def _scan(self):
        self._full_idx = (self.tokens == BOS_ID).nonzero(as_tuple=True)[0].to(torch.int64).cpu().numpy()
        self._ready.set()

    def _maybe_switch(self):
        if self.bos_idx is not self._full_idx and self._ready.is_set():
            self._loader_thread.join()
            self.bos_idx = self._full_idx

    def next_batch(self, num_tokens_local: int, max_seq_len: int):
        self._maybe_switch()
        n = len(self.bos_idx)
        starts = [[] for _ in range(self.world_size)]
        ends = [[] for _ in range(self.world_size)]
        idx = self.i
        for r in range(self.world_size):
            cur_len = 0
            while cur_len <= num_tokens_local:
                if idx >= n:
                    raise StopIteration("Insufficient BOS ahead")
                cur = self.bos_idx[idx]
                starts[r].append(cur)
                end = min(self.bos_idx[idx + 1] if idx + 1 < n else self.size,
                          cur + max_seq_len, cur + num_tokens_local - cur_len + 1)
                ends[r].append(end)
                cur_len += end - cur
                idx += 1
            assert cur_len == num_tokens_local + 1
        self.i = idx
        return starts, ends

    @staticmethod
    def load_async(file: Path, world_size: int = 1):
        result = {}
        ready = threading.Event()
        def load():
            tokens = _load_data_shard(file)
            result['shard'] = Shard(tokens, world_size)
            ready.set()
        thread = threading.Thread(target=load)
        thread.start()
        def get():
            ready.wait()
            thread.join()
            return result['shard']
        return get


def distributed_data_generator(filename_pattern: str, num_tokens: int, max_seq_len: int,
                                grad_accum_steps: int = 1, align_to_bos: bool = True):
    r = dist.get_rank() if dist.is_initialized() else 0
    ws = dist.get_world_size() if dist.is_initialized() else 1
    assert num_tokens % (ws * grad_accum_steps) == 0
    num_tokens = num_tokens // grad_accum_steps

    files = [Path(f) for f in sorted(glob.glob(filename_pattern))]
    if not files:
        raise FileNotFoundError(f"No files found for pattern: {filename_pattern}")

    file_iter = iter(files)
    tokens = _load_data_shard(next(file_iter))
    if align_to_bos:
        shard = Shard(tokens, ws)
        next_shard_getter = Shard.load_async(next(file_iter), ws)
    else:
        pos = 0

    while True:
        num_tokens_local = num_tokens // ws
        max_num_docs = next_multiple_of_n(num_tokens_local // 300, n=128)

        if align_to_bos:
            try:
                seq_starts, seq_ends = shard.next_batch(num_tokens_local, max_seq_len)
                start_idxs, end_idxs = torch.tensor(seq_starts[r]), torch.tensor(seq_ends[r])
            except StopIteration:
                shard = next_shard_getter()
                tokens = shard.tokens
                try:
                    next_shard_getter = Shard.load_async(next(file_iter), ws)
                except StopIteration:
                    next_shard_getter = None
                continue
            buf = torch.cat([tokens[i:j] for i, j in zip(start_idxs, end_idxs)])
            _inputs = buf[:-1]
            _targets = buf[1:]
            end_idxs[-1] -= 1
            cum_lengths = (end_idxs - start_idxs).cumsum(0)
        else:
            if pos + num_tokens + 1 >= len(tokens):
                tokens, pos = _load_data_shard(next(file_iter)), 0
            pos_local = pos + r * num_tokens_local
            buf = tokens[pos_local:pos_local + num_tokens_local + 1]
            _inputs = buf[:-1].view(num_tokens_local)
            _targets = buf[1:].view(num_tokens_local)
            cum_lengths = torch.nonzero(_inputs == BOS_ID)[:, 0]
            pos += num_tokens

        _cum_lengths = torch.full((max_num_docs,), num_tokens_local)
        _cum_lengths[0] = 0
        _cum_lengths[1:len(cum_lengths) + 1] = cum_lengths

        _inputs = _inputs.to(dtype=torch.int32)
        _targets = _targets.to(dtype=torch.int64)
        _cum_lengths = _cum_lengths.to(dtype=torch.int32)

        yield (
            _inputs.to(device="cuda", non_blocking=True),
            _targets.to(device="cuda", non_blocking=True),
            _cum_lengths.to(device="cuda", non_blocking=True),
        )


# -----------------------------------------------------------------------------
# Hyperparameters and schedule

@dataclass
class TeacherHyperparameters:
    data_path: str = os.environ.get("DATA_PATH", ".")
    train_files: str = os.path.join(os.environ.get("DATA_PATH", "."), "data/fineweb10B/fineweb_train_*.bin")
    val_files: str = os.path.join(os.environ.get("DATA_PATH", "."), "data/fineweb10B/fineweb_val_*.bin")
    val_tokens: int = 10485760
    val_batch_size: int = 4 * 64 * 1024 * 8
    # architecture
    num_layers: int = int(os.environ.get("NUM_LAYERS", 3))
    num_heads: int = int(os.environ.get("NUM_HEADS", 6))
    head_dim: int = int(os.environ.get("HEAD_DIM", 128))
    # schedule
    num_iterations: int = int(os.environ.get("NUM_ITERATIONS", 250))
    batch_size: int = int(os.environ.get("BATCH_SIZE", str(8 * 2048 * 8)))
    warmup_frac: float = float(os.environ.get("WARMUP_FRAC", "0.05"))
    # logging and checkpoints
    run_id: str = f"{uuid.uuid4()}"
    val_loss_every: int = int(os.environ.get("VAL_LOSS_EVERY", 25))
    save_checkpoint: bool = bool(int(os.environ.get("SAVE_CHECKPOINT", 1)))
    checkpoint_dir: str = os.environ.get("CHECKPOINT_DIR", "checkpoints/teacher")

hparams = TeacherHyperparameters()
train_max_seq_len = 896  # fixed for teacher (always in W0)


def get_lr(step: int) -> float:
    """LR schedule: linear warmup then cosine decay to 0.1."""
    warmup_steps = max(1, int(hparams.warmup_frac * hparams.num_iterations))
    if step < warmup_steps:
        return step / warmup_steps
    t = (step - warmup_steps) / (hparams.num_iterations - warmup_steps)
    return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * t))


def get_muon_momentum(step: int, warmup_steps=300, cooldown_steps=50, mom_min=0.85, mom_max=0.95):
    cd_start = hparams.num_iterations - cooldown_steps
    if step < warmup_steps:
        return mom_min + (step / warmup_steps) * (mom_max - mom_min)
    elif step > cd_start:
        return mom_max - ((step - cd_start) / cooldown_steps) * (mom_max - mom_min)
    return mom_max


# -----------------------------------------------------------------------------
# Training manager (simplified)

class TrainingManager:
    def __init__(self, model):
        self.model = model

        self.param_table = {
            "attn":       {"optim": "normuon", "comms": "sharded",    "adam_betas": None},
            "mlp":        {"optim": "normuon", "comms": "sharded",    "adam_betas": None},
            "scalars":    {"optim": "adam",    "comms": "replicated", "adam_betas": [0.9,  0.99], "lr_mul": 5.0,  "wd_mul": 0.0},
            "x0_lambdas": {"optim": "adam",    "comms": "replicated", "adam_betas": [0.65, 0.95], "lr_mul": 5.0,  "wd_mul": 0.0},
            "lm_head":    {"optim": "adam",    "comms": "sharded",    "adam_betas": [0.5,  0.95], "wd_mul": 150.},
            "embed":      {"optim": "adam",    "comms": "sharded",    "adam_betas": [0.5,  0.95], "wd_mul": 150.},
        }

        self.work_order = ["scalars", "x0_lambdas", "lm_head", "embed", "attn", "mlp"]

        self.optimizer = NorMuonAndAdam(
            model.named_parameters(),
            param_table=self.param_table,
            scatter_order=list(self.param_table.keys()),
            work_order=self.work_order,
            adam_defaults=dict(lr=0.008, eps=1e-10, weight_decay=0.005),
            normuon_defaults=dict(lr=0.023, momentum=0.95, beta2=0.95, weight_decay=1.2),
        )

        # Split embed from lm_head at 2/3 of training
        self.split_step = (2 * hparams.num_iterations // 3) | 1

    def step_optimizers(self, step: int):
        step_lr = get_lr(step)
        muon_mom = get_muon_momentum(step)
        do_adam = (step % 2 == 1)

        for param, p_cfg in self.optimizer.param_cfgs.items():
            p_cfg.lr = p_cfg.initial_lr * step_lr
            if p_cfg.optim == "normuon":
                p_cfg.momentum = muon_mom

        self.optimizer.step(do_adam=do_adam)

        if step == self.split_step:
            self.optimizer.copy_lm_state_to_embed()

    def reset(self, state=None):
        if state is not None:
            self.optimizer.load_state_dict(state)
        self.optimizer.reset()
        self.model.yarn.reset()

    def get_state(self):
        return copy.deepcopy(self.optimizer.state_dict())


# -----------------------------------------------------------------------------
# int main

logfile = None
if master_process:
    run_id = hparams.run_id
    os.makedirs("logs/dev", exist_ok=True)
    logfile = f"logs/dev/{run_id}.txt"
    print(logfile)

def print0(s, console=False):
    if master_process:
        with open(logfile, "a") as f:
            if console:
                print(s)
            print(s, file=f)

print0(code)
print0("=" * 100)
print0(f"Running Python {sys.version}")
print0(f"Running PyTorch {torch.version.__version__} compiled for CUDA {torch.version.cuda}")
print0(f"Running Triton version {triton.__version__}")

def nvidia_smi():
    import subprocess
    return subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True).stdout
print0(nvidia_smi())
print0("=" * 100)

model_dim = hparams.num_heads * hparams.head_dim
print0(f"Teacher config: {hparams.num_layers}L / {hparams.num_heads}H / {hparams.head_dim}D / {model_dim}dim / {hparams.num_iterations} steps", console=True)

model: nn.Module = GPT(
    vocab_size=50257,
    num_layers=hparams.num_layers,
    num_heads=hparams.num_heads,
    head_dim=hparams.head_dim,
    model_dim=model_dim,
    max_seq_len=hparams.val_batch_size // (grad_accum_steps * world_size),
).cuda()

for m in model.modules():
    if isinstance(m, (nn.Embedding, nn.Linear)):
        m.weight.data = m.weight.data.bfloat16()
model.attn_bank.data = model.attn_bank.data.bfloat16()
model.mlp_bank.data = model.mlp_bank.data.bfloat16()
for param in model.parameters():
    dist.broadcast(param.detach(), 0)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
print0(f"Total parameters: {total_params:,} ({total_params/1e6:.1f}M)", console=True)

model: nn.Module = torch.compile(model, dynamic=False, fullgraph=True)
training_manager = TrainingManager(model)

########################################
#            Warmup kernels            #
######################################## print0("Compiling model and warming up kernels...", console=True)
initial_state = dict(model=copy.deepcopy(model.state_dict()),
                     optimizer=training_manager.get_state())

# Warmup: run a few train + val steps to trigger compilation for all batch sizes
warmup_val_loader = distributed_data_generator(hparams.val_files, hparams.val_batch_size, -1, grad_accum_steps=grad_accum_steps, align_to_bos=False)

warmup_step = 0
warmup_train_loader = distributed_data_generator(hparams.train_files, hparams.batch_size, train_max_seq_len, grad_accum_steps=grad_accum_steps)
for _ in range(2):
    model.eval()
    with torch.no_grad():
        inputs, targets, cum_seqlens = next(warmup_val_loader)
        model(inputs, targets, cum_seqlens)
    model.train()
    for idx in range(grad_accum_steps):
        inputs, targets, cum_seqlens = next(warmup_train_loader)
        loss = model(inputs, targets, cum_seqlens) * grad_scale
        loss.backward()
        del loss
    training_manager.step_optimizers(warmup_step)
    warmup_step += 1
del warmup_train_loader

print0("Resetting model after warmup", console=True)
model.zero_grad(set_to_none=True)
model.load_state_dict(initial_state["model"])
training_manager.reset(initial_state["optimizer"])
del warmup_val_loader, initial_state
model.train()

########################################
#        Training and validation       #
########################################
train_loader = distributed_data_generator(hparams.train_files, hparams.batch_size, train_max_seq_len, grad_accum_steps=grad_accum_steps)

gc.collect()
best_val_loss = float('inf')
training_time_ms = 0
torch.cuda.synchronize()
t0 = time.perf_counter()

train_steps = hparams.num_iterations
for step in range(train_steps + 1):
    last_step = (step == train_steps)

    # --------------- VALIDATION SECTION -----------------
    if last_step or (hparams.val_loss_every > 0 and step % hparams.val_loss_every == 0):
        torch.cuda.synchronize()
        training_time_ms += 1000 * (time.perf_counter() - t0)
        model.eval()
        assert hparams.val_tokens % hparams.val_batch_size == 0
        val_steps = grad_accum_steps * hparams.val_tokens // hparams.val_batch_size
        val_loader = distributed_data_generator(hparams.val_files, hparams.val_batch_size, -1, grad_accum_steps=grad_accum_steps, align_to_bos=False)
        val_loss = 0
        with torch.no_grad():
            for _ in range(val_steps):
                inputs, targets, cum_seqlens = next(val_loader)
                val_loss += model(inputs, targets, cum_seqlens)
        val_loss /= val_steps
        del val_loader
        dist.reduce(val_loss, 0, op=dist.ReduceOp.AVG)

        step_lr = get_lr(step)
        print0(f"step:{step}/{train_steps} val_loss:{val_loss:.4f} train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms/max(step, 1):.2f}ms lr_mul:{step_lr:.4f} muon_lr:{step_lr*0.023:.6f} adam_lr:{step_lr*0.008:.6f}", console=True)

        # Save best checkpoint
        if master_process and hparams.save_checkpoint and val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs(hparams.checkpoint_dir, exist_ok=True)
            ckpt = dict(
                step=step, val_loss=float(val_loss),
                model=model.state_dict(),
                config=dict(num_layers=hparams.num_layers, num_heads=hparams.num_heads,
                            head_dim=hparams.head_dim, model_dim=model_dim, vocab_size=50257),
            )
            torch.save(ckpt, os.path.join(hparams.checkpoint_dir, "state_best.pt"))
            print0(f"  -> Saved best checkpoint (val_loss={val_loss:.4f})", console=True)

        model.train()
        torch.cuda.synchronize()
        t0 = time.perf_counter()

    if last_step:
        # Save final checkpoint
        if master_process and hparams.save_checkpoint:
            os.makedirs(hparams.checkpoint_dir, exist_ok=True)
            ckpt = dict(
                step=step, val_loss=float(val_loss),
                model=model.state_dict(),
                config=dict(num_layers=hparams.num_layers, num_heads=hparams.num_heads,
                            head_dim=hparams.head_dim, model_dim=model_dim, vocab_size=50257),
            )
            torch.save(ckpt, os.path.join(hparams.checkpoint_dir, f"state_step{step:06d}.pt"))
            print0(f"  -> Saved final checkpoint at step {step}", console=True)
        break

    # --------------- TRAINING SECTION -----------------
    for idx in range(grad_accum_steps):
        inputs, targets, cum_seqlens = next(train_loader)
        loss = model(inputs, targets, cum_seqlens) * grad_scale
        loss.backward()
        del loss
    training_manager.step_optimizers(step)

    # logging
    approx_training_time_ms = training_time_ms + 1000 * (time.perf_counter() - t0)
    step_lr = get_lr(step)
    print0(f"step:{step+1}/{train_steps} train_time:{approx_training_time_ms:.0f}ms step_avg:{approx_training_time_ms/(step + 1):.2f}ms lr_mul:{step_lr:.4f} muon_lr:{step_lr*0.023:.6f} adam_lr:{step_lr*0.008:.6f}", console=True)

print0(f"peak memory allocated: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB "
       f"reserved: {torch.cuda.max_memory_reserved() // 1024 // 1024} MiB", console=True)
print0(f"Best val_loss: {best_val_loss:.4f}", console=True)
dist.destroy_process_group()
