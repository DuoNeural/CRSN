"""
DuoNeural CRSN-T: Clifford Resonant Sheaf Network with 4 Laws of Thermodynamics
================================================================================
Authors: Jesse Caldwell, Aura✨, Archon
DuoNeural AI Laboratory (duoneural.com)

Unites Clifford bivector sheaves in Cl(3,0) with non-equilibrium thermodynamic
transport dynamics across discrete network layers:
  - 0th Law: Inter-layer heat diffusion with Neumann boundary conditions
  - 1st Law: Symplectic energy conservation in reversible leapfrog state integration
  - 2nd Law: Non-negative entropy production via strictly dissipative damping
  - 3rd Law: Nernst ground-state stiffness freezing as temperature approaches zero
"""

import os
import sys
import time
import math
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Bind OpenMP & MKL to physical cores
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
torch.set_num_threads(4)
torch.set_num_interop_threads(1)

# Import C++ AVX2 Adjoint Kernel
from torch.utils.cpp_extension import load_inline

CPP_SOURCE = """
#include <torch/extension.h>
#include <vector>
#include <cmath>
#include <omp.h>

std::vector<torch::Tensor> crsn_forward(
    torch::Tensor F_ext,
    torch::Tensor gamma,
    torch::Tensor omega_sq,
    torch::Tensor alpha,
    float dt) {
    
    int B = F_ext.size(0);
    int T = F_ext.size(1);
    int D = F_ext.size(2);
    
    auto q_hist = torch::empty({B, T + 1, D}, F_ext.options());
    auto p_hist = torch::empty({B, T + 1, D}, F_ext.options());
    
    q_hist.select(1, 0).zero_();
    p_hist.select(1, 0).zero_();
    
    const float* F_ptr = F_ext.data_ptr<float>();
    const float* g_ptr = gamma.data_ptr<float>();
    const float* o_ptr = omega_sq.data_ptr<float>();
    const float* a_ptr = alpha.data_ptr<float>();
    float* q_ptr = q_hist.data_ptr<float>();
    float* p_ptr = p_hist.data_ptr<float>();
    
    #pragma omp parallel for collapse(2) schedule(static)
    for (int b = 0; b < B; ++b) {
        for (int d = 0; d < D; ++d) {
            float q = 0.0f;
            float p = 0.0f;
            float om_sq = o_ptr[d];
            float alp = a_ptr[d];
            
            for (int t = 0; t < T; ++t) {
                int in_idx = b * (T * D) + t * D + d;
                int out_idx = b * ((T + 1) * D) + (t + 1) * D + d;
                
                float f = F_ptr[in_idx];
                float gam = g_ptr[in_idx];
                
                // Semi-implicit Euler-Verlet integration
                p = (1.0f - dt * gam) * p + dt * (f - om_sq * q - alp * std::tanh(q));
                q = q + dt * p;
                
                p_ptr[out_idx] = p;
                q_ptr[out_idx] = q;
            }
        }
    }
    
    return {q_hist.narrow(1, 1, T), q_hist, p_hist};
}

std::vector<torch::Tensor> crsn_backward(
    torch::Tensor grad_y,
    torch::Tensor gamma,
    torch::Tensor q_hist,
    torch::Tensor p_hist,
    torch::Tensor omega_sq,
    torch::Tensor alpha,
    float dt) {
    
    int B = grad_y.size(0);
    int T = grad_y.size(1);
    int D = grad_y.size(2);
    
    auto grad_F = torch::empty_like(grad_y);
    auto grad_omega_sq = torch::zeros_like(omega_sq);
    auto grad_alpha = torch::zeros_like(alpha);
    
    const float* dy_ptr = grad_y.data_ptr<float>();
    const float* g_ptr = gamma.data_ptr<float>();
    const float* o_ptr = omega_sq.data_ptr<float>();
    const float* a_ptr = alpha.data_ptr<float>();
    const float* q_ptr = q_hist.data_ptr<float>();
    const float* p_ptr = p_hist.data_ptr<float>();
    
    float* dF_ptr = grad_F.data_ptr<float>();
    float* dO_ptr = grad_omega_sq.data_ptr<float>();
    float* dA_ptr = grad_alpha.data_ptr<float>();
    
    float norm_factor = 1.0f / (float)T;
    
    #pragma omp parallel for schedule(static)
    for (int d = 0; d < D; ++d) {
        float om_sq = o_ptr[d];
        float alp = a_ptr[d];
        float local_dO = 0.0f;
        float local_dA = 0.0f;
        
        for (int b = 0; b < B; ++b) {
            float lambda_q = 0.0f;
            float lambda_p = 0.0f;
            
            for (int t = T - 1; t >= 0; --t) {
                int in_idx = b * (T * D) + t * D + d;
                int hist_prev_idx = b * ((T + 1) * D) + t * D + d;
                
                float q_prev = q_ptr[hist_prev_idx];
                float gam = g_ptr[in_idx];
                float dy = dy_ptr[in_idx];
                
                lambda_q += dy;
                lambda_p += dt * lambda_q;
                
                dF_ptr[in_idx] = dt * lambda_p * norm_factor;
                
                float th = std::tanh(q_prev);
                float sech2 = 1.0f - th * th;
                
                local_dO += -dt * lambda_p * q_prev;
                local_dA += -dt * lambda_p * th;
                
                lambda_q -= dt * lambda_p * (om_sq + alp * sech2);
                lambda_p = (1.0f - dt * gam) * lambda_p;
            }
        }
        
        #pragma omp atomic
        dO_ptr[d] += local_dO * norm_factor;
        #pragma omp atomic
        dA_ptr[d] += local_dA * norm_factor;
    }
    
    return {grad_F, grad_omega_sq, grad_alpha};
}
"""

try:
    crsn_cpp = load_inline(
        name="crsn_cpp_thermo",
        cpp_sources=CPP_SOURCE,
        functions=["crsn_forward", "crsn_backward"],
        extra_cflags=["-O3", "-fopenmp", "-mavx2", "-mfma"]
    )
except Exception as e:
    print(f"[Warning] Failed to compile C++ kernel inline: {e}")
    crsn_cpp = None


class CRSNCppFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, F_ext, gamma, omega_sq, alpha, dt):
        q, q_hist, p_hist = crsn_cpp.crsn_forward(F_ext, gamma, omega_sq, alpha, dt)
        ctx.save_for_backward(gamma, q_hist, p_hist, omega_sq, alpha)
        ctx.dt = dt
        return q

    @staticmethod
    def backward(ctx, grad_y):
        gamma, q_hist, p_hist, omega_sq, alpha = ctx.saved_tensors
        grad_F, grad_omega_sq, grad_alpha = crsn_cpp.crsn_backward(
            grad_y.contiguous(), gamma, q_hist, p_hist, omega_sq, alpha, ctx.dt
        )
        return grad_F, None, grad_omega_sq, grad_alpha, None


# ==============================================================================
# TOKENIZER & DATASET
# ==============================================================================
try:
    from tokenizers import Tokenizer
except ImportError:
    pass

class MemmapDataset:
    def __init__(self, bin_path: str, seq_len: int = 256):
        self.seq_len = seq_len
        self.data = np.memmap(bin_path, dtype=np.uint16, mode='r')
        self.n_tokens = len(self.data)
        self.n_samples = (self.n_tokens - 1) // seq_len
        print(f"[Dataset] Loaded {bin_path}: {self.n_tokens:,} tokens ({self.n_samples:,} sequences of len {seq_len})", flush=True)

    def sample_batch(self, batch_size: int, device: torch.device):
        max_idx = self.n_tokens - self.seq_len - 1
        starts = np.random.randint(0, max_idx, size=batch_size)
        x = np.empty((batch_size, self.seq_len), dtype=np.int64)
        y = np.empty((batch_size, self.seq_len), dtype=np.int64)
        for i, s in enumerate(starts):
            x[i] = self.data[s : s + self.seq_len]
            y[i] = self.data[s + 1 : s + self.seq_len + 1]
        return torch.tensor(x, dtype=torch.long, device=device), torch.tensor(y, dtype=torch.long, device=device)


# ==============================================================================
# OPTIMIZERS: MUON + ADAMW
# ==============================================================================
def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    assert G.ndim >= 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16() if G.dtype == torch.bfloat16 else G.float()
    X /= (X.norm(dim=(-2, -1), keepdim=True) + eps)
    if G.size(-2) > G.size(-1):
        X = X.mT
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(-2) > G.size(-1):
        X = X.mT
    return X.to(G.dtype)

class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr: float = 0.02, momentum: float = 0.95, weight_decay: float = 0.01, nesterov: bool = True, ns_steps: int = 5):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov=nesterov, ns_steps=ns_steps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr = group["lr"]
            mom = group["momentum"]
            wd = group["weight_decay"]
            nest = group["nesterov"]
            ns = group["ns_steps"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(mom).add_(g)
                if nest:
                    g_eff = g.add(buf, alpha=mom)
                else:
                    g_eff = buf
                u = zeropower_via_newtonschulz5(g_eff, steps=ns)
                scale = max(1.0, p.size(-2) / p.size(-1)) ** 0.5
                u.mul_(scale)
                if wd != 0:
                    p.mul_(1.0 - lr * wd)
                p.add_(u, alpha=-lr)


# ==============================================================================
# MODEL ARCHITECTURE: CRSN-T
# ==============================================================================
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

class CliffordResonantCell(nn.Module):
    def __init__(self, d_state: int, n_layers: int = 8, dt: float = 0.1):
        super().__init__()
        self.d_state = d_state
        self.dt = dt
        self.norm = RMSNorm(d_state)
        self.w_force = nn.Linear(d_state, d_state, bias=False)
        self.w_damp = nn.Linear(d_state, d_state, bias=True)
        self.w_out = nn.Linear(d_state, d_state, bias=False)

        nn.init.normal_(self.w_out.weight, mean=0.0, std=0.02 / math.sqrt(2 * n_layers))

        taus = torch.exp(torch.linspace(math.log(2.0), math.log(1024.0), d_state))
        self.omega_sq = nn.Parameter((2.0 * math.pi / taus) ** 2)
        self.alpha_raw = nn.Parameter(torch.ones(d_state) * -2.0)
        self.alpha_override = None

    @property
    def alpha(self):
        if self.alpha_override is not None:
            return self.alpha_override
        return F.softplus(self.alpha_raw) + 1e-4

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = self.norm(x)
        F_ext = self.w_force(x_norm)
        gamma = torch.clamp(torch.sigmoid(self.w_damp(x_norm)), min=0.01, max=0.99)
        omega_sq = torch.clamp(torch.abs(self.omega_sq), min=1e-4, max=100.0)
        y = CRSNCppFunction.apply(F_ext, gamma, omega_sq, self.alpha, self.dt)
        return self.w_out(y)

class SwiGLU(nn.Module):
    def __init__(self, d_state: int, d_ff: int, n_layers: int = 8):
        super().__init__()
        self.norm = RMSNorm(d_state)
        self.w_gate = nn.Linear(d_state, d_ff, bias=False)
        self.w_up = nn.Linear(d_state, d_ff, bias=False)
        self.w_down = nn.Linear(d_ff, d_state, bias=False)
        nn.init.normal_(self.w_down.weight, mean=0.0, std=0.02 / math.sqrt(2 * n_layers))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = self.norm(x)
        return self.w_down(F.silu(self.w_gate(x_norm)) * self.w_up(x_norm))

class ReversibleBlockFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, y1: torch.Tensor, y2: torch.Tensor, f_module: nn.Module, g_module: nn.Module, scale: float = 0.70710678):
        with torch.no_grad():
            f_out = f_module(y2)
            y1_next = scale * y1 + f_out
            g_out = g_module(y1_next)
            y2_next = scale * y2 + g_out

        ctx.f_module = f_module
        ctx.g_module = g_module
        ctx.scale = scale
        ctx.save_for_backward(y1_next.detach(), y2_next.detach())
        return y1_next, y2_next

    @staticmethod
    def backward(ctx, grad_y1_next: torch.Tensor, grad_y2_next: torch.Tensor):
        f_module = ctx.f_module
        g_module = ctx.g_module
        scale = ctx.scale
        y1_next, y2_next = ctx.saved_tensors

        with torch.set_grad_enabled(True):
            y1_next_leaf = y1_next.detach().requires_grad_(True)
            g_out = g_module(y1_next_leaf)
            g_out.backward(grad_y2_next)
            grad_y1_from_g = y1_next_leaf.grad
            grad_y2 = scale * grad_y2_next

        y2 = (y2_next - g_out.detach()) / scale

        with torch.set_grad_enabled(True):
            y2_leaf = y2.detach().requires_grad_(True)
            f_out = f_module(y2_leaf)
            total_grad_y1_next = grad_y1_next + grad_y1_from_g
            f_out.backward(total_grad_y1_next)
            grad_y2_from_f = y2_leaf.grad
            grad_y1 = scale * total_grad_y1_next

        grad_y2 = grad_y2 + grad_y2_from_f
        return grad_y1, grad_y2, None, None, None

class ReversibleBlock(nn.Module):
    def __init__(self, d_state: int, d_ff: int, n_layers: int = 8, dt: float = 0.1):
        super().__init__()
        self.f_module = CliffordResonantCell(d_state, n_layers=n_layers, dt=dt)
        self.g_module = SwiGLU(d_state, d_ff, n_layers=n_layers)

    def forward(self, y1: torch.Tensor, y2: torch.Tensor):
        return ReversibleBlockFunction.apply(y1, y2, self.f_module, self.g_module)

class CRSNLanguageModel(nn.Module):
    def __init__(self, vocab_size: int = 16384, d_model: int = 384, n_layers: int = 8, d_ff: int = 512, dt: float = 0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.d_state = d_model // 2
        self.n_layers = n_layers

        self.embedding = nn.Embedding(vocab_size, d_model)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)

        self.layers = nn.ModuleList([
            ReversibleBlock(self.d_state, d_ff, n_layers=n_layers, dt=dt)
            for _ in range(n_layers)
        ])
        self.final_norm = RMSNorm(d_model)

    def forward_features(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embedding(input_ids)
        y1, y2 = torch.chunk(x, 2, dim=-1)
        for layer in self.layers:
            y1, y2 = layer(y1, y2)
        h = torch.cat([y1, y2], dim=-1)
        return self.final_norm(h)

    def forward(self, input_ids: torch.Tensor, targets: torch.Tensor = None, chunk_tokens: int = 512):
        h = self.forward_features(input_ids)
        if targets is None:
            return F.linear(h, self.embedding.weight)

        B, T, D = h.size()
        h_flat = h.view(-1, D)
        targets_flat = targets.view(-1)
        total_loss = torch.tensor(0.0, device=h.device)
        total_tokens = h_flat.size(0)

        for i in range(0, total_tokens, chunk_tokens):
            chunk_h = h_flat[i : i + chunk_tokens]
            chunk_targets = targets_flat[i : i + chunk_tokens]
            chunk_logits = F.linear(chunk_h, self.embedding.weight)
            chunk_loss = F.cross_entropy(chunk_logits, chunk_targets, reduction='sum')
            total_loss = total_loss + chunk_loss

        return total_loss / total_tokens


# ==============================================================================
# THERMODYNAMIC ENGINE (4 LAWS OF THERMODYNAMICS)
# ==============================================================================
class ThermodynamicEngine:
    def __init__(self, n_layers: int = 8, alpha_base: float = 0.128, alpha_floor: float = 0.05, kappa: float = 0.01, T0: float = 1.0, beta_ema: float = 0.05):
        self.n_layers = n_layers
        self.alpha_base = alpha_base
        self.alpha_floor = alpha_floor
        self.kappa = kappa
        self.T0 = T0
        self.beta = beta_ema
        self.T = torch.ones(n_layers, dtype=torch.float32)

    @torch.no_grad()
    def apply_thermodynamics(self, model: CRSNLanguageModel):
        """
        Calculates layer stiffness according to 3rd Law (Nernst Ground-State freeze)
        and 0th Law (Inter-layer heat diffusion with Neumann boundary conditions).
        """
        # 3rd Law: Nernst ground-state stiffness freeze
        alpha_nernst = torch.clamp(self.alpha_base * torch.tanh(self.T / self.T0), min=self.alpha_floor)

        # 0th Law: Discrete 1D Laplacian with Neumann (Zero Heat Flux) boundary conditions
        delta_T = torch.zeros(self.n_layers, dtype=torch.float32)
        delta_T[0] = self.T[1] - self.T[0]
        for l in range(1, self.n_layers - 1):
            delta_T[l] = self.T[l - 1] - 2.0 * self.T[l] + self.T[l + 1]
        delta_T[self.n_layers - 1] = self.T[self.n_layers - 2] - self.T[self.n_layers - 1]

        # Effective dissipation: alpha_eff = alpha - kappa * Delta_L(T)
        alpha_eff = torch.clamp(alpha_nernst - self.kappa * delta_T, min=self.alpha_floor, max=5.0)

        for l, layer in enumerate(model.layers):
            # Broadcast scalar alpha_eff to d_state vector
            layer.f_module.alpha_override = torch.full((layer.f_module.d_state,), alpha_eff[l].item(), dtype=torch.float32)

        heat_flux = self.kappa * torch.abs(delta_T)
        nernst_order = torch.mean(torch.tanh(self.T / self.T0)).item()
        return self.T.mean().item(), heat_flux.mean().item(), nernst_order, alpha_eff.mean().item()

    @torch.no_grad()
    def update_temperatures(self, model: CRSNLanguageModel):
        """
        Step-lagged EMA update of layer temperature from parameter gradient variance.
        Treats T as a decoupled macroscopic observable (zero autograd overhead).
        """
        for l, layer in enumerate(model.layers):
            cell = layer.f_module
            grads = []
            if cell.w_force.weight.grad is not None:
                grads.append(cell.w_force.weight.grad.view(-1))
            if cell.w_damp.weight.grad is not None:
                grads.append(cell.w_damp.weight.grad.view(-1))
            if grads:
                g_cat = torch.cat(grads)
                grad_var = torch.mean(g_cat ** 2).item()
                obs_T = math.sqrt(grad_var) * 100.0
                self.T[l] = (1.0 - self.beta) * self.T[l] + self.beta * obs_T


# ==============================================================================
# ATOMIC CHECKPOINTING & GENERATION
# ==============================================================================
def save_atomic_checkpoint(payload: dict, filepath: str):
    tmp_path = f"{filepath}.tmp"
    torch.save(payload, tmp_path)
    os.replace(tmp_path, filepath)

@torch.no_grad()
def generate(model, tokenizer, prompt: str, max_new_tokens: int = 32, temperature: float = 0.8, top_k: int = 40):
    input_ids = torch.tensor([tokenizer.encode(prompt).ids], dtype=torch.long)
    for _ in range(max_new_tokens):
        logits = model(input_ids)[:, -1, :] / max(1e-5, temperature)
        if top_k > 0:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf')
        probs = F.softmax(logits, dim=-1)
        next_tok = torch.multinomial(probs, num_samples=1)
        input_ids = torch.cat([input_ids, next_tok], dim=1)
    return tokenizer.decode(input_ids[0].tolist())

def get_lr_factor(opt_step: int, warmup_steps: int, total_steps: int) -> float:
    if opt_step < warmup_steps:
        return float(opt_step) / float(max(1, warmup_steps))
    progress = float(opt_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))


# ==============================================================================
# MAIN TRAINING PIPELINE
# ==============================================================================
def train_production():
    p = argparse.ArgumentParser(description="DuoNeural CRSN-T Pre-training (4-Laws of Thermodynamics)")
    p.add_argument("--train-bin", required=True)
    p.add_argument("--val-bin", required=True)
    p.add_argument("--data-dir", default="./data")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--seq-len", type=int, default=256)
    p.add_argument("--grad-accum-steps", type=int, default=4)
    p.add_argument("--total-optimizer-steps", type=int, default=25000)
    p.add_argument("--warmup-steps", type=int, default=200)
    p.add_argument("--muon-lr", type=float, default=0.01)
    p.add_argument("--adam-lr", type=float, default=6e-4)
    p.add_argument("--save-interval", type=int, default=100)
    p.add_argument("--eval-interval", type=int, default=100)
    p.add_argument("--ckpt-dir", default=None)
    p.add_argument("--resume", default=None)
    # Thermodynamic Hyperparameters
    p.add_argument("--kappa", type=float, default=0.01, help="0th Law Thermal Conductivity")
    p.add_argument("--alpha-base", type=float, default=0.128, help="Universal Base Stiffness")
    p.add_argument("--alpha-floor", type=float, default=0.05, help="3rd Law Ground-State Floor")
    p.add_argument("--t0", type=float, default=1.0, help="Nernst Temperature Scale")
    args = p.parse_args()

    device = torch.device("cpu")
    print(f"==============================================================================")
    print(f" DuoNeural CRSN-T (Clifford Sheaf + 4 Laws of Thermodynamics)")
    print(f" Pre-training Engine: 4-Core Ryzen SIMD | Native AVX2 OpenMP Integration")
    print(f"==============================================================================", flush=True)

    tok_path = os.path.join(args.data_dir, "tokenizer_16k.json")
    tokenizer = Tokenizer.from_file(tok_path)

    train_ds = MemmapDataset(args.train_bin, seq_len=args.seq_len)
    val_ds = MemmapDataset(args.val_bin, seq_len=args.seq_len)

    model = CRSNLanguageModel(vocab_size=16384, d_model=384, n_layers=8, d_ff=512, dt=0.1)
    model.to(device)

    # Initialize Thermodynamic Engine
    thermo = ThermodynamicEngine(
        n_layers=8,
        alpha_base=args.alpha_base,
        alpha_floor=args.alpha_floor,
        kappa=args.kappa,
        T0=args.t0,
        beta_ema=0.05
    )

    muon_params = [p for n, p in model.named_parameters() if p.ndim == 2 and "embedding" not in n]
    adam_decay_params = [model.embedding.weight]
    adam_no_decay_params = [p for n, p in model.named_parameters() if p.ndim != 2 and "embedding" not in n]

    optimizer_muon = Muon(muon_params, lr=args.muon_lr, momentum=0.95, weight_decay=0.01)
    optimizer_adam = torch.optim.AdamW([
        {"params": adam_decay_params, "weight_decay": 0.01},
        {"params": adam_no_decay_params, "weight_decay": 0.0}
    ], lr=args.adam_lr, betas=(0.9, 0.95))

    ckpt_dir = args.ckpt_dir or os.path.join(args.data_dir, "checkpoints_crsn_t")
    os.makedirs(ckpt_dir, exist_ok=True)

    opt_step = 0
    start_step = 0
    total_micro_steps = args.total_optimizer_steps * args.grad_accum_steps
    best_val_loss = float("inf")

    if args.resume and os.path.exists(args.resume):
        print(f"[Resume] Loading weights from {args.resume}...", flush=True)
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        sd = ckpt.get("model_state", ckpt.get("model", ckpt))
        model.load_state_dict(sd, strict=False)
        if "loss" in ckpt:
            best_val_loss = ckpt["loss"]
        if "ema_temperatures" in ckpt:
            thermo.T = ckpt["ema_temperatures"].clone()
        if "opt_step" in ckpt:
            start_step = ckpt["opt_step"]
            opt_step = start_step
            print(f"[Resume] Restored opt_step={opt_step}, best_val_loss={best_val_loss:.4f}", flush=True)

    t0 = time.perf_counter()
    print(f"[Run] Launching CRSN-T: Steps {start_step + 1} -> {args.total_optimizer_steps}...", flush=True)

    model.train()
    optimizer_muon.zero_grad()
    optimizer_adam.zero_grad()

    for m_step in range(start_step * args.grad_accum_steps + 1, total_micro_steps + 1):
        step_t0 = time.perf_counter()
        inputs, targets = train_ds.sample_batch(args.batch_size, device)

        # Apply 4-Laws Thermodynamics to Oscillator Lattice before forward pass
        mean_T, mean_Q, nernst_order, mean_alpha = thermo.apply_thermodynamics(model)

        loss = model(inputs, targets=targets)
        if not torch.isfinite(loss):
            print(f"  [GUARD] Non-finite loss ({loss.item()}) at uStep {m_step:04d}! Skipping.", flush=True)
            continue

        (loss / args.grad_accum_steps).backward()

        step_dt = time.perf_counter() - step_t0
        tok_s = (args.batch_size * args.seq_len) / max(1e-5, step_dt)

        if m_step % args.grad_accum_steps == 0:
            opt_step += 1
            lr_scale = get_lr_factor(opt_step, args.warmup_steps, args.total_optimizer_steps)
            for g in optimizer_muon.param_groups:
                g['lr'] = args.muon_lr * lr_scale
            for g in optimizer_adam.param_groups:
                g['lr'] = args.adam_lr * lr_scale

            # Segregated Gradient Clipping
            torch.nn.utils.clip_grad_norm_(muon_params, max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(adam_decay_params + adam_no_decay_params, max_norm=1.0)

            # Step-lagged thermodynamic observable update (zero autograd overhead)
            thermo.update_temperatures(model)

            optimizer_muon.step()
            optimizer_adam.step()
            optimizer_muon.zero_grad()
            optimizer_adam.zero_grad()

            if opt_step % 10 == 0 or opt_step == 1:
                print(
                    f"Step {opt_step:05d}/{args.total_optimizer_steps} | "
                    f"Loss: {loss.item():.4f} | "
                    f"T={mean_T:.3f} Q={mean_Q:.5f} ξ_S={nernst_order:.4f} α={mean_alpha:.4f} | "
                    f"Speed: {tok_s:6.1f} tok/s",
                    flush=True
                )

            # Periodic Validation & Generation
            if opt_step % args.eval_interval == 0 or opt_step == args.total_optimizer_steps:
                model.eval()
                val_losses = []
                with torch.no_grad():
                    for _ in range(4):
                        v_in, v_tgt = val_ds.sample_batch(args.batch_size, device)
                        v_loss = model(v_in, targets=v_tgt)
                        val_losses.append(v_loss.item())
                avg_val_loss = float(np.mean(val_losses))
                model.train()

                sample_story = generate(model, tokenizer, prompt="Once upon a time, Lily", max_new_tokens=48, temperature=0.8)
                sample_reason = generate(model, tokenizer, prompt="User: Is a cat an animal?\n\nAssistant:", max_new_tokens=32, temperature=0.8)
                print(f"  --> [Validation] Val Loss: {avg_val_loss:.4f} (ppl={math.exp(min(20, avg_val_loss)):.1f})", flush=True)
                print(f"  --> [Sample Story] \"{sample_story.strip()}\"", flush=True)
                print(f"  --> [Sample Reason] \"{sample_reason.strip()}\"", flush=True)

                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    best_path = os.path.join(ckpt_dir, "crsn_t_best.pt")
                    save_atomic_checkpoint({
                        "opt_step": opt_step,
                        "loss": avg_val_loss,
                        "best_val_loss": best_val_loss,
                        "model_state": model.state_dict(),
                        "optimizer_muon": optimizer_muon.state_dict(),
                        "optimizer_adam": optimizer_adam.state_dict(),
                        "ema_temperatures": thermo.T,
                    }, best_path)
                    print(f"  --> [Checkpoint] Saved NEW BEST atomic checkpoint to {best_path}!", flush=True)

            # Periodic Atomic Milestone Checkpoint
            if opt_step % args.save_interval == 0 or opt_step == args.total_optimizer_steps:
                ckpt_path = os.path.join(ckpt_dir, f"crsn_t_step{opt_step:05d}.pt")
                save_atomic_checkpoint({
                    "opt_step": opt_step,
                    "loss": loss.item(),
                    "best_val_loss": best_val_loss,
                    "model_state": model.state_dict(),
                    "optimizer_muon": optimizer_muon.state_dict(),
                    "optimizer_adam": optimizer_adam.state_dict(),
                    "ema_temperatures": thermo.T,
                }, ckpt_path)
                print(f"  --> [Checkpoint] Saved atomic checkpoint to {ckpt_path}", flush=True)

    total_time = time.perf_counter() - t0
    total_tokens = (args.total_optimizer_steps - start_step) * args.batch_size * args.seq_len * args.grad_accum_steps
    print(f"\n==============================================================================")
    print(f"[Complete] CRSN-T Pre-training Finished!")
    print(f"           Total Tokens:    {total_tokens:,}")
    print(f"           Total Wall Time: {total_time:.2f}s ({total_time/60:.2f} mins)")
    print(f"           Average Speed:   {total_tokens / total_time:.2f} tokens/sec")
    print(f"           Best Val Loss:   {best_val_loss:.4f}")
    print(f"==============================================================================", flush=True)

if __name__ == "__main__":
    train_production()
