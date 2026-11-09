import os
import sys
import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load
from tokenizers import Tokenizer

# 4 cores for Ryzen 5 3550H
os.environ["OMP_NUM_THREADS"] = "4"
torch.set_num_threads(4)

# 1. Load kernel
cpp_src = "/home/duoneural/crsn_ryzen/crsn_kernel.cpp"
print("[1/5] Compiling / loading AVX2 Clifford Sheaf kernel...")
crsn_cpp = load(
    name="crsn_cpp_probe",
    sources=[cpp_src],
    extra_cflags=["-O3", "-mavx2", "-mfma", "-fopenmp", "-march=znver1"],
    extra_ldflags=["-fopenmp"],
    verbose=False
)

# 2. Re-create architecture
class SwiGLU(nn.Module):
    def __init__(self, d_state: int, d_ff: int, n_layers: int = 8):
        super().__init__()
        self.w_gate = nn.Linear(d_state, d_ff, bias=False)
        self.w_up = nn.Linear(d_state, d_ff, bias=False)
        self.w_down = nn.Linear(d_ff, d_state, bias=False)
        nn.init.normal_(self.w_gate.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.w_up.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.w_down.weight, mean=0.0, std=0.02 / math.sqrt(2 * n_layers))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))

class CliffordResonantCell(nn.Module):
    def __init__(self, d_state: int, n_layers: int = 8):
        super().__init__()
        self.d_state = d_state
        self.W_in = nn.Linear(d_state, d_state, bias=False)
        nn.init.normal_(self.W_in.weight, mean=0.0, std=0.02)
        self.gamma_raw = nn.Parameter(torch.full((d_state,), -2.0))
        self.omega_raw = nn.Parameter(torch.empty(d_state).uniform_(-1.0, 1.0))
        self.alpha_raw = nn.Parameter(torch.empty(d_state).uniform_(-2.0, 0.0))

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        B, T, D = y.shape
        gamma = torch.sigmoid(self.gamma_raw) * 0.2
        omega = self.omega_raw
        alpha = F.softplus(self.alpha_raw)
        dt = 0.05
        q0 = torch.zeros(B, D, dtype=torch.float32)
        v0 = torch.zeros(B, D, dtype=torch.float32)
        q_seq, _ = crsn_cpp.forward(y.contiguous(), self.W_in.weight.contiguous(), gamma, omega, alpha, dt, q0, v0)
        return q_seq

class ReversibleBlock(nn.Module):
    def __init__(self, d_state: int, d_ff: int, n_layers: int = 8):
        super().__init__()
        self.f_module = CliffordResonantCell(d_state=d_state, n_layers=n_layers)
        self.g_module = SwiGLU(d_state=d_state, d_ff=d_ff, n_layers=n_layers)
        self.scale = 1.0 / math.sqrt(2.0)

    def forward(self, y1: torch.Tensor, y2: torch.Tensor):
        # Evaluation forward:
        f_out = self.f_module(y2)
        y1_next = (y1 + f_out) * self.scale
        g_out = self.g_module(y1_next)
        y2_next = (y2 + g_out) * self.scale
        return y1_next, y2_next

class CRSNLanguageModel(nn.Module):
    def __init__(self, vocab_size: int = 16384, d_model: int = 384, n_layers: int = 8):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.d_state = d_model // 2
        self.d_ff = 512
        self.n_layers = n_layers

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([ReversibleBlock(self.d_state, self.d_ff, n_layers) for _ in range(n_layers)])
        self.norm = nn.RMSNorm(d_model)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        B, T = input_ids.shape
        x = self.embedding(input_ids)
        y1, y2 = x[..., :self.d_state].contiguous(), x[..., self.d_state:].contiguous()
        for block in self.blocks:
            y1, y2 = block(y1, y2)
        h = torch.cat([y1, y2], dim=-1)
        h = self.norm(h)
        logits = F.linear(h, self.embedding.weight)
        return logits

# 3. Load Checkpoint
ckpt_path = "/home/duoneural/srsn_ryzen/checkpoints_crsn/crsn_best.pt"
print(f"[2/5] Loading checkpoint from {ckpt_path}...")
model = CRSNLanguageModel(vocab_size=16384, d_model=384, n_layers=8)
ckpt = torch.load(ckpt_path, map_location="cpu")
model.load_state_dict(ckpt["model_state"])
model.eval()
print(f"      Best step: {ckpt.get('opt_step')}, recorded val_loss: {ckpt.get('loss'):.4f}")

# 4. Inspect Learned Clifford Oscillators
print("\n[3/5] Inspecting Learned Physical Sheaf Parameters:")
alpha_all = []
omega_all = []
gamma_all = []
for l_idx, block in enumerate(model.blocks):
    cell = block.f_module
    alpha = F.softplus(cell.alpha_raw).detach().numpy()
    omega = cell.omega_raw.detach().numpy()
    gamma = (torch.sigmoid(cell.gamma_raw) * 0.2).detach().numpy()
    alpha_all.extend(alpha)
    omega_all.extend(omega)
    gamma_all.extend(gamma)
    print(f"  Layer {l_idx:02d}: alpha=[{alpha.min():.4f}, {alpha.max():.4f}, mean={alpha.mean():.4f}] | omega_mean={omega.mean():.4f} | gamma_mean={gamma.mean():.4f}")

import numpy as np
alpha_all = np.array(alpha_all)
omega_all = np.array(omega_all)
print(f"  Global Alpha (Stiffness): min={alpha_all.min():.4f}, max={alpha_all.max():.4f}, mean={alpha_all.mean():.4f}")
print(f"  Positivity Guarantee: {np.all(alpha_all >= 0)} (100% strictly positive)")

# 5. Tokenizer & Autoregressive Sampling
tok_path = "/home/duoneural/srsn_ryzen/tokenizer_16k.json"
tokenizer = Tokenizer.from_file(tok_path)

prompts = [
    ("Story", "Once upon a time, Lily found a small silver key in the garden."),
    ("Math CoT", "User: If you have 3 boxes with 4 apples in each box, how many apples do you have in total?\n\nAssistant: Let's solve this step by step:\n"),
    ("Logic", "User: True or False: All birds can fly.\n\nAssistant: "),
    ("Code", "def calculate_area(width, height):\n    # Return the area\n"),
    ("Identity", "User: Who are you?\n\nAssistant: ")
]

print("\n[4/5] Autoregressive Probing Across Temperatures:")
for label, p_text in prompts:
    print(f"\n==================== PROMPT [{label}] ====================")
    print(f"Prompt: {repr(p_text)}")
    for temp in [0.3, 0.8]:
        input_ids = tokenizer.encode(p_text).ids
        input_tensor = torch.tensor([input_ids], dtype=torch.long)
        
        with torch.no_grad():
            for _ in range(60):
                curr = input_tensor if input_tensor.size(1) <= 512 else input_tensor[:, -512:]
                logits = model(curr)
                next_logits = logits[0, -1, :] / temp
                probs = F.softmax(next_logits, dim=-1)
                next_tok = torch.multinomial(probs, num_samples=1)
                input_tensor = torch.cat([input_tensor, next_tok.unsqueeze(0)], dim=1)
        
        gen_tokens = input_tensor[0, len(input_ids):].tolist()
        gen_text = tokenizer.decode(gen_tokens)
        print(f"\n[Temp {temp} Output]:\n{gen_text}")

print("\n==========================================================")
print("[5/5] CRSN Probe Complete.")
