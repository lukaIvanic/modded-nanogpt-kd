"""
Teacher model for knowledge distillation inference.
Extracted from train_gpt_teacher.py — eval-only (no training ops needed).

Usage:
    from teacher_model import load_teacher
    teacher = load_teacher("checkpoints/teacher/state_best.pt", device)
"""

import os
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from triton_kernels import FusedLinearReLUSquareFunction


def norm(x: Tensor):
    return F.rms_norm(x, (x.size(-1),))


def next_multiple_of_n(v: float | int, *, n: int):
    return next(x for x in range(n, int(v) + 1 + n, n) if x >= v)


class Yarn(nn.Module):
    def __init__(self, head_dim, max_seq_len, device):
        super().__init__()
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self._device = device
        self.reset()

    def rotary(self, x_BTHD):
        assert self.factor1.size(0) >= x_BTHD.size(-3)
        factor1 = self.factor1[None, :x_BTHD.size(-3), None, :]
        factor2 = self.factor2[None, :x_BTHD.size(-3), None, :]
        x_flip = x_BTHD.view(*x_BTHD.shape[:-1], x_BTHD.shape[-1] // 2, 2).flip(-1).view(x_BTHD.shape)
        return factor1 * x_BTHD + factor2 * x_flip

    def reset(self):
        angular_freq = (1 / 1024) ** torch.linspace(0, 1, steps=self.head_dim // 4, dtype=torch.float32, device=self._device)
        angular_freq = angular_freq.repeat_interleave(2)
        angular_freq = torch.cat([angular_freq, angular_freq.new_zeros(self.head_dim // 2)])
        t = torch.arange(2 * self.max_seq_len, dtype=torch.float32, device=self._device)
        theta = torch.outer(t, angular_freq)
        self.factor1 = nn.Buffer(theta.cos().to(torch.bfloat16), persistent=False)
        self.factor2 = nn.Buffer(theta.sin().to(torch.bfloat16), persistent=False)
        self.factor2[..., 1::2] *= -1
        self.angular_freq = angular_freq
        self.attn_scale = 0.1


_flash_attn_interface = None

def set_flash_attn_interface(interface):
    """Set the flash_attn_interface (call before using teacher model)."""
    global _flash_attn_interface
    _flash_attn_interface = interface


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
        y = _flash_attn_interface.flash_attn_varlen_func(
            q[0], k[0], v[0],
            cu_seqlens_q=seqlens, cu_seqlens_k=seqlens,
            max_seqlen_q=max_len, max_seqlen_k=max_len,
            causal=True, softmax_scale=yarn.attn_scale,
            window_size=(-1, -1))
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


class CastedLinearT(nn.Module):
    """Linear with transposed weight storage. Eval-only: no FP8."""
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(in_features, out_features, dtype=torch.bfloat16))

    def forward(self, x):
        return x @ self.weight.type_as(x)


class TeacherGPT(nn.Module):
    def __init__(self, vocab_size: int, num_layers: int, num_heads: int, head_dim: int, model_dim: int, max_seq_len: int, device='cuda'):
        super().__init__()
        self.num_layers = num_layers
        self.vocab_size = next_multiple_of_n(vocab_size, n=128)

        hdim = num_heads * head_dim
        mlp_hdim = 4 * model_dim

        # Attention weight bank — world_size=1 for inference
        self.attn_bank = nn.Parameter(torch.empty(num_layers, 4 * model_dim, hdim))
        self.mlp_bank = nn.Parameter(torch.empty(num_layers, 2, mlp_hdim, model_dim))

        self.blocks = nn.ModuleList([
            Block(model_dim, head_dim, num_heads) for _ in range(num_layers)
        ])
        self.yarn = Yarn(head_dim, max_seq_len, device)

        self.lm_head = CastedLinearT(model_dim, self.vocab_size)
        self.embed = nn.Embedding(self.vocab_size, model_dim)

        self.x0_lambdas = nn.Parameter(torch.zeros(num_layers))
        num_scalars = 3 * num_layers
        self.scalars = nn.Parameter(torch.zeros(num_scalars))

    def forward(self, input_seq: Tensor, cum_seqlens: Tensor, max_len: int):
        """Returns softcapped logits in bfloat16. Shape: (1, seq_len, vocab_size)."""
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
        logits = self.lm_head(x)  # bfloat16
        logits = 23 * torch.sigmoid((logits + 5) / 7.5)
        return logits

    @torch.no_grad()
    def get_loss(self, input_seq: Tensor, target_seq: Tensor, cum_seqlens: Tensor, max_len: int) -> Tensor:
        """Compute teacher cross-entropy loss (sanity check). Logits in bf16, CE in float."""
        logits = self.forward(input_seq, cum_seqlens, max_len)
        return F.cross_entropy(logits.float().view(-1, logits.size(-1)), target_seq, reduction="mean")

    def get_kd_loss(self, input_seq: Tensor, student_logits: Tensor, cum_seqlens: Tensor, max_len: int, temperature: float = 1.0) -> Tensor:
        """Compute KD loss (KL divergence) between student and teacher.

        Teacher forward runs in no_grad; KL computation preserves student gradients.
        Logits stay in bfloat16, KL computation uses float32.

        Args:
            input_seq: token ids (1D)
            student_logits: student's softcapped logits (1, seq_len, vocab_size), bfloat16
            cum_seqlens: cumulative sequence lengths
            max_len: max sequence length for flash attention
            temperature: KD temperature (higher = softer)

        Returns:
            Scalar KD loss (with grad through student_logits)
        """
        with torch.no_grad():
            teacher_logits = self.forward(input_seq, cum_seqlens, max_len)

        # Flatten to (seq_len, vocab_size)
        t = (teacher_logits.view(-1, teacher_logits.size(-1)) / temperature).float()
        s = (student_logits.view(-1, student_logits.size(-1)) / temperature).float()

        kd_loss = F.kl_div(
            F.log_softmax(s, dim=-1),
            F.log_softmax(t, dim=-1),
            log_target=True,
            reduction='batchmean',
        ) * (temperature ** 2)

        return kd_loss


def load_teacher(checkpoint_path: str, device: torch.device, max_seq_len: int = 4096) -> TeacherGPT:
    """Load a pretrained teacher model from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ckpt['config']
    teacher = TeacherGPT(
        vocab_size=config['vocab_size'],
        num_layers=config['num_layers'],
        num_heads=config['num_heads'],
        head_dim=config['head_dim'],
        model_dim=config['model_dim'],
        max_seq_len=max_seq_len,
        device=device,
    )

    # The checkpoint was saved with world_size padding on attn_bank/mlp_bank.
    # Our model may have different shapes. Use strict=False and handle manually.
    state = ckpt['model']

    # Strip _orig_mod. prefix added by torch.compile
    state = {k.removeprefix('_orig_mod.'): v for k, v in state.items()}
    teacher.load_state_dict(state, strict=True)

    teacher.to(device=device, dtype=torch.bfloat16)
    teacher.eval()
    teacher.requires_grad_(False)
    return teacher
