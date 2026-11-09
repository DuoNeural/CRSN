#!/usr/bin/env python3
"""
CRSN (Clifford Resonant Sheaf Network) Production Pretraining Daemon
Hardware Target: AMD Ryzen 5 3550H (4 Cores / Zen+ AVX2 SIMD)
Architecture:
  - Clifford Resonant Sheaf Coupling (8 Layers, D=384, D_state=192, D_ff=512)
  - Reversible RevNet Trunk with Contractive Phase-Space Adjoint
  - Softplus Stiffness Floor: K(q) >= 1e-4 (Eliminates negative bifurcation)
  - Sequence-Normalized Adjoint Integration (1/T)
  - Dual Optimizer: Muon (2D matrices) + Segregated AdamW (Embeddings & Resonance Bank)
"""
import os
import sys
import time
import math
import psutil
import resource
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load
from tokenizers import Tokenizer

# ==============================================================================
# 0. HARDWARE THREAD PINNING (AMD RYZEN 5 3550H: 4 PHYSICAL CORES)
# ==============================================================================
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
torch.set_num_threads(4)
torch.set_num_interop_threads(1)

def get_peak_rss_mb():
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return usage / 1024.0

def get_current_rss_mb():
    return psutil.Process().memory_info().rss / (1024.0 * 1024.0)

# ==============================================================================
# 1. LOAD NATIVE C++ OPENMP / AVX2 SIMD KERNEL
# ==============================================================================
current_dir = os.path.dirname(os.path.abspath(__file__))
cpp_src = os.path.join(current_dir, "crsn_kernel.cpp")

crsn_cpp = load(
    name="crsn_cpp",
    sources=[cpp_src],
    extra_cflags=["-O3", "-march=native", "-mavx2", "-mfma", "-fopenmp", "-ffast-math", "-fno-math-errno"],
    extra_ldflags=["-fopenmp"],
    verbose=False
)

class CRSNCppFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, F_ext, gamma, omega_sq, alpha, dt):
        outputs, q_hist, p_hist = crsn_cpp.forward(F_ext, gamma, omega_sq, alpha, float(dt))
        ctx.save_for_backward(gamma, q_hist, p_hist, omega_sq, alpha)
        ctx.dt = dt
        return outputs

    @staticmethod
    def backward(ctx, grad_out):
        gamma, q_hist, p_hist, omega_sq, alpha = ctx.saved_tensors
        dF_ext, dgamma, domega_sq, dalpha = crsn_cpp.backward(
            grad_out, gamma, q_hist, p_hist, omega_sq, alpha, float(ctx.dt)
        )
        return dF_ext, dgamma, domega_sq, dalpha, None

# ==============================================================================
# 2. ZERO-COPY UINT16 MEMMAP DATASTREAMER
# ==============================================================================
class MemmapDataset:
    def __init__(self, bin_path: str, seq_len: int = 256):
        if not os.path.exists(bin_path):
            raise FileNotFoundError(f"Binary file not found: {bin_path}. Run prepare_data.py first!")
        self.data = np.memmap(bin_path, dtype=np.uint16, mode='r')
        self.seq_len = seq_len
        self.num_tokens = len(self.data)
        if self.num_tokens <= seq_len + 1:
            raise ValueError(f"Dataset in {bin_path} has only {self.num_tokens} tokens, need > {seq_len + 1}")

    def sample_batch(self, batch_size: int, device: torch.device):
        ix = np.random.randint(0, self.num_tokens - self.seq_len - 1, size=batch_size)
        batch = np.stack([self.data[i : i + self.seq_len + 1].astype(np.int64) for i in ix])
        t_batch = torch.from_numpy(batch).to(device)
        return t_batch[:, :-1].contiguous(), t_batch[:, 1:].contiguous()

# ==============================================================================
# 3. OPTIMIZER: NEWTON-SCHULZ ORTHOGONALIZED MOMENTUM (MUON)
# ==============================================================================
def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    assert G.ndim == 2
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.float()
    norm = X.norm() + eps
    X = X / norm
    transposed = X.size(0) > X.size(1)
    if transposed:
        X = X.T
    for _ in range(steps):
        A = torch.mm(X, X.T)
        B = b * A + c * torch.mm(A, A)
        X = a * X + torch.mm(B, X)
    if transposed:
        X = X.T
    return X.to(dtype=G.dtype)

class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr: float = 0.02, momentum: float = 0.95, nesterov: bool = True, weight_decay: float = 0.0):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            nesterov = group['nesterov']
            weight_decay = group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue
                g = p.grad
                if weight_decay != 0:
                    g = g.add(p, alpha=weight_decay)

                state = self.state[p]
                if 'momentum_buffer' not in state:
                    buf = state['momentum_buffer'] = torch.zeros_like(g)
                else:
                    buf = state['momentum_buffer']

                buf.mul_(momentum).add_(g)
                if nesterov:
                    g = g.add(buf, alpha=momentum)
                else:
                    g = buf

                u = zeropower_via_newtonschulz5(g)
                p.data.add_(u, alpha=-lr)

# ==============================================================================
# 4. ARCHITECTURE: CLIFFORD RESONANT SHEAF NETWORK (CRSN)
# ==============================================================================
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * self.weight

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
        # Softplus parameterization guarantees alpha >= 0 (no saddle bifurcation)
        self.alpha_raw = nn.Parameter(torch.ones(d_state) * -2.0)

    @property
    def alpha(self):
        return F.softplus(self.alpha_raw) + 1e-4

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = self.norm(x)
        F_ext = self.w_force(x_norm)
        # Bounded non-zero dissipation
        gamma = torch.clamp(torch.sigmoid(self.w_damp(x_norm)), min=0.01, max=0.99)
        # Bounded stiffness spectrum protecting against resonance singularities
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
        ctx.f_module = f_module
        ctx.g_module = g_module
        ctx.scale = scale
        with torch.no_grad():
            f_out = f_module(y2) * scale
            y1_next = y1 + f_out
            g_out = g_module(y1_next) * scale
            y2_next = y2 + g_out
        ctx.save_for_backward(y1_next.detach(), y2_next.detach())
        return y1_next, y2_next

    @staticmethod
    def backward(ctx, dy1_next: torch.Tensor, dy2_next: torch.Tensor):
        y1_next, y2_next = ctx.saved_tensors
        f_module = ctx.f_module
        g_module = ctx.g_module
        scale = ctx.scale

        # Reconstruct y2 and backprop through G
        with torch.enable_grad():
            y1_next_eval = y1_next.detach().requires_grad_(True)
            g_eval = g_module(y1_next_eval) * scale
            g_eval.backward(dy2_next)
            dy1_from_g = y1_next_eval.grad

        y2 = y2_next - g_eval.detach()
        dy1 = dy1_next + dy1_from_g

        # Reconstruct y1 and backprop through F
        with torch.enable_grad():
            y2_eval = y2.detach().requires_grad_(True)
            f_eval = f_module(y2_eval) * scale
            f_eval.backward(dy1)
            dy2_from_f = y2_eval.grad

        y1 = y1_next - f_eval.detach()
        dy2 = dy2_next + dy2_from_f
        return dy1, dy2, None, None, None

class ReversibleBlock(nn.Module):
    def __init__(self, d_state: int, d_ff: int, n_layers: int = 8):
        super().__init__()
        self.f_module = CliffordResonantCell(d_state=d_state, n_layers=n_layers)
        self.g_module = SwiGLU(d_state=d_state, d_ff=d_ff, n_layers=n_layers)
        self.scale = 1.0 / math.sqrt(2.0) # 0.7071 residual scale

    def forward(self, y1: torch.Tensor, y2: torch.Tensor):
        return ReversibleBlockFunction.apply(y1, y2, self.f_module, self.g_module, self.scale)

class CRSNLanguageModel(nn.Module):
    def __init__(self, vocab_size: int = 16384, d_model: int = 384, n_layers: int = 8):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.d_state = d_model // 2 # 192 dims = 24 x 8 Clifford bivectors
        self.d_ff = 512            # Exact balanced SwiGLU ratio
        self.n_layers = n_layers

        self.embedding = nn.Embedding(vocab_size, d_model)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
        self.layers = nn.ModuleList([
            ReversibleBlock(d_state=self.d_state, d_ff=self.d_ff, n_layers=n_layers)
            for _ in range(n_layers)
        ])
        self.final_norm = RMSNorm(d_model)

    def forward_features(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embedding(input_ids)
        y1, y2 = torch.chunk(x, chunks=2, dim=-1)
        for layer in self.layers:
            y1, y2 = layer(y1, y2)
        out = torch.cat([y1, y2], dim=-1)
        return self.final_norm(out)

    def forward(self, input_ids: torch.Tensor, targets: torch.Tensor = None, chunk_tokens: int = 512):
        out = self.forward_features(input_ids)
        if targets is None:
            return F.linear(out, self.embedding.weight)

        flat_out = out.view(-1, self.d_model)
        flat_targets = targets.view(-1)
        total_tokens = flat_out.size(0)
        total_loss = 0.0
        for start in range(0, total_tokens, chunk_tokens):
            end = min(start + chunk_tokens, total_tokens)
            logits_chunk = F.linear(flat_out[start:end], self.embedding.weight)
            loss_chunk = F.cross_entropy(logits_chunk, flat_targets[start:end], reduction='sum')
            total_loss = total_loss + loss_chunk
        return total_loss / total_tokens

# ==============================================================================
# 5. AUTOREGRESSIVE GENERATION
# ==============================================================================
@torch.no_grad()
def generate(model: CRSNLanguageModel, tokenizer, prompt: str, max_new_tokens: int = 48, temperature: float = 0.8, top_k: int = 40):
    model.eval()
    input_ids = tokenizer.encode(prompt).ids
    input_tensor = torch.tensor([input_ids], dtype=torch.long)
    
    for _ in range(max_new_tokens):
        curr_input = input_tensor if input_tensor.size(1) <= 512 else input_tensor[:, -512:]
        logits = model(curr_input)
        next_token_logits = logits[:, -1, :] / max(1e-5, temperature)
        
        if top_k > 0:
            v, _ = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
            next_token_logits[next_token_logits < v[:, [-1]]] = -float('Inf')
            
        probs = F.softmax(next_token_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        input_tensor = torch.cat([input_tensor, next_token], dim=1)
        
        if next_token.item() == tokenizer.token_to_id("</s>"):
            break
            
    model.train()
    generated_ids = input_tensor[0].tolist()
    return tokenizer.decode(generated_ids)

# ==============================================================================
# 6. PRODUCTION TRAINING LOOP
# ==============================================================================
def get_lr_factor(step, warmup_steps, total_steps, min_ratio=0.1):
    if step < warmup_steps:
        return max(1e-4, float(step) / float(max(1, warmup_steps)))
    progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    progress = min(1.0, max(0.0, progress))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_ratio + (1.0 - min_ratio) * cosine

def train_production():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="/home/duoneural/srsn_ryzen")
    p.add_argument("--train-bin", default=None)
    p.add_argument("--val-bin", default=None)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--seq-len", type=int, default=256)
    p.add_argument("--grad-accum-steps", type=int, default=4)
    p.add_argument("--total-optimizer-steps", type=int, default=2500)
    p.add_argument("--warmup-steps", type=int, default=25)
    p.add_argument("--muon-lr", type=float, default=0.02)
    p.add_argument("--adam-lr", type=float, default=1e-3)
    p.add_argument("--save-interval", type=int, default=25)
    p.add_argument("--eval-interval", type=int, default=25)
    args = p.parse_args()

    device = torch.device("cpu")
    print(f"\n==============================================================================", flush=True)
    print(f"[Init] CRSN Production Pre-Training (AMD Ryzen 5 3550H / Zen+ AVX2)", flush=True)
    print(f"       Batch={args.batch_size}, SeqLen={args.seq_len}, Accum={args.grad_accum_steps}", flush=True)
    print(f"       Tokens/OptStep: {args.batch_size * args.seq_len * args.grad_accum_steps:,}", flush=True)
    print(f"==============================================================================", flush=True)

    tok_path = os.path.join(args.data_dir, "tokenizer_16k.json")
    
    # Resolve train/val dataset paths
    train_bin = args.train_bin or ("/dev/shm/train_golden.bin" if os.path.exists("/dev/shm/train_golden.bin") else os.path.join(args.data_dir, "train_golden.bin"))
    val_bin = args.val_bin or ("/dev/shm/val_golden.bin" if os.path.exists("/dev/shm/val_golden.bin") else os.path.join(args.data_dir, "val_golden.bin"))

    if not os.path.exists(tok_path):
        raise FileNotFoundError(f"Tokenizer not found at {tok_path}!")
    tokenizer = Tokenizer.from_file(tok_path)
    vocab_size = tokenizer.get_vocab_size()

    print(f"[Data] Training Dataset:   {train_bin} ({os.path.getsize(train_bin)/(1024*1024):.2f} MB)")
    print(f"[Data] Validation Dataset: {val_bin} ({os.path.getsize(val_bin)/1024:.2f} KB)")
    train_ds = MemmapDataset(train_bin, seq_len=args.seq_len)
    val_ds = MemmapDataset(val_bin, seq_len=args.seq_len)
    print(f"[Data] Loaded memmap dataset: Train={train_ds.num_tokens:,} tokens | Val={val_ds.num_tokens:,} tokens", flush=True)

    model = CRSNLanguageModel(vocab_size=vocab_size, d_model=384, n_layers=8).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    embed_params = sum(p.numel() for p in model.embedding.parameters())
    layer_params = sum(p.numel() for p in model.layers.parameters())
    print(f"[Model] 16k Vocab CRSN-Compact Parameter Breakdown:")
    print(f"        - Embedding Table ({vocab_size} x 384): {embed_params / 1e6:.2f}M ({embed_params:,} params)")
    print(f"        - 8 RevNet Trunk Layers:              {layer_params / 1e6:.2f}M ({layer_params:,} params)")
    print(f"        - Total Tied Parameter Footprint:       {total_params / 1e6:.2f}M ({total_params:,} params)", flush=True)

    # Segregate parameter groups
    muon_params = [p for n, p in model.named_parameters() if p.ndim == 2 and "embedding" not in n]
    adam_decay_params = [p for n, p in model.named_parameters() if "embedding" in n]
    adam_no_decay_params = [p for n, p in model.named_parameters() if p.ndim != 2 and "embedding" not in n]

    print(f"[Optim] Parameter Segregation:")
    print(f"        - Muon 2D Matrices (weight_decay=0.01):      {sum(p.numel() for p in muon_params) / 1e6:.2f}M ({len(muon_params)} tensors)")
    print(f"        - AdamW Decay (Embedding, wd=0.01):          {sum(p.numel() for p in adam_decay_params) / 1e6:.2f}M ({len(adam_decay_params)} tensors)")
    print(f"        - AdamW No-Decay (omega_sq, alpha, norms):   {sum(p.numel() for p in adam_no_decay_params):,} params ({len(adam_no_decay_params)} tensors)", flush=True)

    optimizer_muon = Muon(muon_params, lr=args.muon_lr, momentum=0.95, weight_decay=0.01)
    optimizer_adam = torch.optim.AdamW([
        {"params": adam_decay_params, "weight_decay": 0.01},
        {"params": adam_no_decay_params, "weight_decay": 0.0}
    ], lr=args.adam_lr, betas=(0.9, 0.95))

    ckpt_dir = os.path.join(args.data_dir, "checkpoints_crsn")
    os.makedirs(ckpt_dir, exist_ok=True)

    opt_step = 0
    total_micro_steps = args.total_optimizer_steps * args.grad_accum_steps
    best_val_loss = float("inf")

    t0 = time.perf_counter()
    print(f"[Run] Launching CRSN Pretraining: Steps 1 -> {args.total_optimizer_steps} ({total_micro_steps} micro-steps)...", flush=True)

    model.train()
    optimizer_muon.zero_grad()
    optimizer_adam.zero_grad()

    for m_step in range(1, total_micro_steps + 1):
        step_t0 = time.perf_counter()
        inputs, targets = train_ds.sample_batch(args.batch_size, device)

        loss = model(inputs, targets=targets)
        
        if not torch.isfinite(loss):
            print(f"  [GUARD] Non-finite loss ({loss.item()}) at uStep {m_step:04d}! Skipping backward pass.", flush=True)
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
            norm_muon = torch.nn.utils.clip_grad_norm_(muon_params, max_norm=1.0)
            norm_embed = torch.nn.utils.clip_grad_norm_(adam_decay_params, max_norm=1.0)
            norm_reson = torch.nn.utils.clip_grad_norm_(adam_no_decay_params, max_norm=0.5)

            comp_norm = math.sqrt(norm_muon.item()**2 + norm_embed.item()**2 + norm_reson.item()**2)

            if not math.isfinite(comp_norm):
                print(f"  [GUARD] Non-finite GradNorm ({comp_norm:.3f}) at Step {opt_step:03d}! Skipping update.", flush=True)
                optimizer_muon.zero_grad()
                optimizer_adam.zero_grad()
                continue

            optimizer_muon.step()
            optimizer_adam.step()
            optimizer_muon.zero_grad()
            optimizer_adam.zero_grad()

            rss = get_current_rss_mb()
            print(f"Step {opt_step:03d}/{args.total_optimizer_steps:03d} (uStep {m_step:04d}) | Loss: {loss.item():.4f} | GradNorm: {comp_norm:.3f} (M:{norm_muon.item():.2f}, E:{norm_embed.item():.2f}, R:{norm_reson.item():.2f}) | LR: {lr_scale:.3f} | Speed: {tok_s:6.1f} tok/s | RSS: {rss:5.1f} MB", flush=True)

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

                # Save best checkpoint
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    best_path = os.path.join(ckpt_dir, "crsn_best.pt")
                    torch.save({"opt_step": opt_step, "loss": avg_val_loss, "model_state": model.state_dict()}, best_path)
                    print(f"  --> [Checkpoint] Saved NEW BEST checkpoint to {best_path}!", flush=True)

            # Periodic Checkpoint
            if opt_step % args.save_interval == 0 or opt_step == args.total_optimizer_steps:
                ckpt_path = os.path.join(ckpt_dir, f"crsn_step{opt_step:04d}.pt")
                torch.save({"opt_step": opt_step, "loss": loss.item(), "model_state": model.state_dict()}, ckpt_path)
                print(f"  --> [Checkpoint] Saved atomic checkpoint to {ckpt_path}", flush=True)

    total_time = time.perf_counter() - t0
    total_tokens = args.total_optimizer_steps * args.batch_size * args.seq_len * args.grad_accum_steps
    print(f"\n==============================================================================")
    print(f"[Complete] CRSN Pre-training Finished!")
    print(f"           Total Tokens:    {total_tokens:,}")
    print(f"           Total Wall Time: {total_time:.2f}s ({total_time/60:.2f} mins)")
    print(f"           Average Speed:   {total_tokens / total_time:.2f} tokens/sec")
    print(f"           Best Val Loss:   {best_val_loss:.4f}")
    print(f"==============================================================================", flush=True)

if __name__ == "__main__":
    train_production()
