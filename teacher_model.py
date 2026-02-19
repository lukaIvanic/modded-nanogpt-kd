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
        """Returns raw (unsoftcapped) logits in bfloat16. Shape: (1, seq_len, vocab_size)."""
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
        logits = self.lm_head(x)  # bfloat16, raw (no softcap for KD)
        return logits

    @torch.no_grad()
    def get_loss(self, input_seq: Tensor, target_seq: Tensor, cum_seqlens: Tensor, max_len: int) -> Tensor:
        """Compute teacher cross-entropy loss (sanity check). Logits in bf16, CE in float."""
        logits = self.forward(input_seq, cum_seqlens, max_len)
        logits = 23 * torch.sigmoid((logits + 5) / 7.5)  # softcap for CE
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

        _debug = not hasattr(self, '_dbg_count')
        if _debug:
            self._dbg_count = 0
        self._dbg_count += 1
        _show = self._dbg_count <= 160  # ~20 steps * 8 accum

        if _show:
            print(f"[DEBUG kd] teacher_logits dtype={teacher_logits.dtype} shape={teacher_logits.shape} absmax={teacher_logits.abs().max().item():.4f} mean={teacher_logits.mean().item():.4f} std={teacher_logits.std().item():.4f}")
            print(f"[DEBUG kd] student_logits dtype={student_logits.dtype} shape={student_logits.shape} absmax={student_logits.abs().max().item():.4f} mean={student_logits.mean().item():.4f} std={student_logits.std().item():.4f}")

        # Flatten to (seq_len, vocab_size)
        t = (teacher_logits.view(-1, teacher_logits.size(-1)) / temperature).float()
        s = (student_logits.view(-1, student_logits.size(-1)) / temperature).float()

        if _show:
            print(f"[DEBUG kd] t dtype={t.dtype} absmax={t.abs().max().item():.4f}")
            print(f"[DEBUG kd] s dtype={s.dtype} absmax={s.abs().max().item():.4f} requires_grad={s.requires_grad}")

        s_log = F.log_softmax(s, dim=-1)
        t_log = F.log_softmax(t, dim=-1)
        if _show:
            print(f"[DEBUG kd] s_log_softmax min={s_log.min().item():.4f} max={s_log.max().item():.4f}")
            print(f"[DEBUG kd] t_log_softmax min={t_log.min().item():.4f} max={t_log.max().item():.4f}")

        kd_loss = F.kl_div(
            s_log,
            t_log,
            log_target=True,
            reduction='sum',
        ) * (temperature ** 2)

        if _show:
            print(f"[DEBUG kd] kd_loss={kd_loss.item():.6f} dtype={kd_loss.dtype} requires_grad={kd_loss.requires_grad}")
        return kd_loss


def apply_kd_dynamic_norm(hard_loss, soft_loss, eps=1e-8):
    """Scale soft_loss to match hard_loss magnitude (per-step, detached)."""
    ratio = hard_loss.detach() / (soft_loss.detach().abs() + eps)
    scale = ratio.clamp_min(0.0)
    return soft_loss * scale


def load_teacher(checkpoint_path: str, device: torch.device, max_seq_len: int = 4096, default_config: dict = None):
    """Load a pretrained teacher model from checkpoint.

    Handles both TeacherGPT checkpoints (with 'config' key) and student GPT checkpoints
    (saved from train_gpt.py). Student checkpoints are detected by extra keys like
    'value_embeds' and loaded via StudentTeacherWrapper.
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = ckpt['model']
    state = {k.removeprefix('_orig_mod.'): v for k, v in state.items()}

    # Detect student checkpoint by presence of student-only keys
    is_student_ckpt = 'value_embeds' in state or 'bigram_embed.weight' in state
    if is_student_ckpt:
        print(f"[load_teacher] Detected student architecture checkpoint, using StudentTeacherWrapper")
        print(f"[load_teacher] Caller must use load_student_as_teacher() instead — pass GPT class to avoid circular import")
        raise ValueError(
            "Student architecture checkpoint detected. Use load_student_as_teacher() from train_gpt.py instead, "
            "passing the GPT class directly to avoid circular imports."
        )

    config = ckpt.get('config', default_config)
    if config is None:
        raise ValueError("Checkpoint has no 'config' key and no default_config provided")
    teacher = TeacherGPT(
        vocab_size=config['vocab_size'],
        num_layers=config['num_layers'],
        num_heads=config['num_heads'],
        head_dim=config['head_dim'],
        model_dim=config['model_dim'],
        max_seq_len=max_seq_len,
        device=device,
    )
    teacher.load_state_dict(state, strict=True)
    teacher.to(device=device, dtype=torch.bfloat16)
    teacher.eval()
    teacher.requires_grad_(False)
    return teacher


class StudentTeacherWrapper(nn.Module):
    """Wraps a student GPT model to provide the teacher interface (get_loss, get_kd_loss).

    The student GPT forward needs (input_seq, target_seq, seqlens, bigram_input_seq, schedule_cfg).
    This wrapper provides a simpler interface matching what the KD training loop expects.
    Raw logits are extracted by calling lm_head directly on the transformer output.
    """
    def __init__(self, gpt_model, attn_args_class):
        super().__init__()
        self.gpt = gpt_model
        self.num_layers = gpt_model.num_layers
        self.vocab_size = gpt_model.vocab_size
        self._AttnArgs = attn_args_class

    def _get_raw_logits(self, input_seq, cum_seqlens, max_len):
        """Run the student GPT transformer and return raw logits (no softcap)."""
        gpt = self.gpt

        # Reconstruct what forward() does, but stop before softcap/loss
        resid_lambdas = gpt.scalars[:gpt.num_layers]
        x0_lambdas = gpt.x0_lambdas
        sa_lambdas = gpt.scalars[gpt.num_layers:3 * gpt.num_layers].view(-1, 2)
        bigram_lambdas = gpt.scalars[3 * gpt.num_layers:4 * gpt.num_layers]
        smear_lambda = gpt.scalars[4 * gpt.num_layers]
        backout_lambda = gpt.scalars[4 * gpt.num_layers + 1]
        skip_lambda = gpt.scalars[4 * gpt.num_layers + 2]

        # Use max window sizes for eval (no schedule needed)
        bm_sizes = [1024, 1024, 1024, max_len, 1024, 1024, None, 1024, 1024, 1024, max_len]
        key_offset = [b == max_len for b in bm_sizes]

        x = gpt.embed(input_seq)
        # Zero bigram contribution for teacher inference (no bigram_input_seq available)
        x0_bigram = torch.zeros(1, x.size(0), x.size(-1), device=x.device, dtype=x.dtype)

        # Value embeddings
        ve = gpt.value_embeds.view(5, gpt.vocab_size, -1)[:, input_seq]
        ve = [None, ve[0], ve[1]] + [None] * (gpt.num_layers - 6) + [ve[2], ve[3], ve[4]]

        # Smear
        smear_gate_out = smear_lambda * torch.sigmoid(gpt.smear_gate(x[1:, :gpt.smear_gate.weight.size(-1)]))
        x = torch.cat([x[:1], x[1:] + smear_gate_out * x[:-1]])
        x = x0 = norm(x[None])

        # Gate banks
        ag = [w.bfloat16() for w in gpt.attn_gate_bank.unbind(0)]
        veg = [w.bfloat16() for w in gpt.ve_gate_bank.unbind(0)]
        attn_gates = ag[:6] + [None] + ag[6:]
        ve_gates = [None, veg[0], veg[1]] + [None] * (gpt.num_layers - 6) + [veg[2], veg[3], veg[4]]

        attn_weights = gpt.attn_bank.unbind(0)
        mlp_fcs = gpt.mlp_bank[:, 0, :, :].unbind(0)
        mlp_projs = gpt.mlp_bank[:, 1, :, :].unbind(0)

        skip_connections = []
        skip_in = [3]
        skip_out = [6]
        x_backout = None
        backout_layer = 7

        for i in range(gpt.num_layers):
            yarn = gpt.yarn_paired_head if i in gpt.paired_head_layers else gpt.yarn
            attn_args = self._AttnArgs(
                ve=ve[i],
                sa_lambdas=sa_lambdas[i],
                seqlens=cum_seqlens,
                bm_size=bm_sizes[i],
                yarn=yarn,
                key_offset=key_offset[i],
                attn_gate_w=attn_gates[i],
                ve_gate_w=ve_gates[i],
                train_max_seq_len=max_len
            )
            if i in skip_out:
                skip_gate_out = torch.sigmoid(skip_lambda) * 2 * torch.sigmoid(gpt.skip_gate(x0[..., :gpt.skip_gate.weight.size(-1)]))
                x = x + skip_gate_out * skip_connections.pop()
            if i == 0:
                x = (resid_lambdas[0] + x0_lambdas[0]) * x + bigram_lambdas[0] * x0_bigram
            else:
                x = resid_lambdas[i] * x + x0_lambdas[i] * x0 + bigram_lambdas[i] * x0_bigram

            qkvo_w = attn_weights[gpt.layer_to_attn_idx[i]] if i in gpt.layer_to_attn_idx else None
            c_fc = mlp_fcs[gpt.layer_to_mlp_idx[i]] if i in gpt.layer_to_mlp_idx else None
            c_proj = mlp_projs[gpt.layer_to_mlp_idx[i]] if i in gpt.layer_to_mlp_idx else None

            x = gpt.blocks[i](x, attn_args, qkvo_w, c_fc, c_proj)
            if i in skip_in:
                skip_connections.append(x)
            if i == backout_layer:
                x_backout = x

        x -= backout_lambda * x_backout
        x = norm(x)
        logits = gpt.lm_head(x)  # raw logits, no softcap
        return logits

    def forward(self, input_seq, cum_seqlens, max_len):
        return self._get_raw_logits(input_seq, cum_seqlens, max_len)

    @torch.no_grad()
    def get_loss(self, input_seq, target_seq, cum_seqlens, max_len):
        logits = self.forward(input_seq, cum_seqlens, max_len)
        logits = 23 * torch.sigmoid((logits + 5) / 7.5)
        return F.cross_entropy(logits.float().view(-1, logits.size(-1)), target_seq, reduction="mean")

    def get_kd_loss(self, input_seq, student_logits, cum_seqlens, max_len, temperature=1.0):
        with torch.no_grad():
            teacher_logits = self.forward(input_seq, cum_seqlens, max_len)

        _debug = not hasattr(self, '_dbg_count')
        if _debug:
            self._dbg_count = 0
        self._dbg_count += 1
        _show = self._dbg_count <= 160

        if _show:
            print(f"[DEBUG kd] teacher_logits dtype={teacher_logits.dtype} shape={teacher_logits.shape} absmax={teacher_logits.abs().max().item():.4f}")
            print(f"[DEBUG kd] student_logits dtype={student_logits.dtype} shape={student_logits.shape} absmax={student_logits.abs().max().item():.4f}")

        t = (teacher_logits.view(-1, teacher_logits.size(-1)) / temperature).float()
        s = (student_logits.view(-1, student_logits.size(-1)) / temperature).float()
        s_log = F.log_softmax(s, dim=-1)
        t_log = F.log_softmax(t, dim=-1)
        kd_loss = F.kl_div(s_log, t_log, log_target=True, reduction='sum') * (temperature ** 2)

        if _show:
            print(f"[DEBUG kd] kd_loss={kd_loss.item():.6f}")
        return kd_loss


def load_student_as_teacher(checkpoint_path, device, max_seq_len, gpt_class, attn_args_class=None, config=None):
    """Load a student GPT checkpoint as teacher via StudentTeacherWrapper.

    Args:
        checkpoint_path: path to checkpoint file
        device: torch device
        max_seq_len: max sequence length for attention
        gpt_class: the GPT class from train_gpt (passed to avoid circular import)
        attn_args_class: the AttnArgs dataclass from train_gpt
        config: model config dict (optional, has defaults)
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = ckpt['model']
    state = {k.removeprefix('_orig_mod.'): v for k, v in state.items()}
    config = config or ckpt.get('config') or dict(vocab_size=50257, num_layers=11, num_heads=6, head_dim=128, model_dim=768)
    gpt = gpt_class(
        vocab_size=config['vocab_size'],
        num_layers=config['num_layers'],
        num_heads=config['num_heads'],
        head_dim=config['head_dim'],
        model_dim=config['model_dim'],
        max_seq_len=max_seq_len,
        kd_mode=False,
    )
    gpt.load_state_dict(state, strict=False)  # strict=False: padding in mlp_bank etc.
    print(f"[load_teacher] Loaded student GPT: {config['num_layers']}L, vocab={config['vocab_size']}")
    wrapper = StudentTeacherWrapper(gpt, attn_args_class)
    wrapper.to(device=device, dtype=torch.bfloat16)
    wrapper.eval()
    wrapper.requires_grad_(False)
    return wrapper
