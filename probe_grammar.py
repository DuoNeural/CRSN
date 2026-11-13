import os
import sys
import argparse
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load
from tokenizers import Tokenizer

parser = argparse.ArgumentParser()
parser.add_argument('--ckpt', type=str, required=True)
parser.add_argument('--label', type=str, default='CRSN')
args = parser.parse_args()

# 2 cores to leave CPU headroom for background training
os.environ['OMP_NUM_THREADS'] = '2'
torch.set_num_threads(2)

cpp_src = '/home/duoneural/crsn_ryzen/crsn_kernel.cpp'
crsn_cpp = load(
    name='crsn_cpp_probe',
    sources=[cpp_src],
    extra_cflags=['-O3', '-mavx2', '-mfma', '-fopenmp', '-march=znver1'],
    extra_ldflags=['-fopenmp'],
    verbose=False
)

class SwiGLU(nn.Module):
    def __init__(self, d_state: int, d_ff: int, n_layers: int = 8):
        super().__init__()
        self.w_gate = nn.Linear(d_state, d_ff, bias=False)
        self.w_up = nn.Linear(d_state, d_ff, bias=False)
        self.w_down = nn.Linear(d_ff, d_state, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))

class CliffordResonantCell(nn.Module):
    def __init__(self, d_state: int, n_layers: int = 8):
        super().__init__()
        self.d_state = d_state
        self.W_in = nn.Linear(d_state, d_state, bias=False)
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

print(f'=== EVALUATING {args.label} ===')
print(f'Loading checkpoint: {args.ckpt}')
model = CRSNLanguageModel(vocab_size=16384, d_model=384, n_layers=8)
ckpt = torch.load(args.ckpt, map_location='cpu')
model.load_state_dict(ckpt['model_state'])
model.eval()
print(f'Step: {ckpt.get("opt_step")}, val_loss: {ckpt.get("loss", "N/A")}')

tok_path = '/home/duoneural/srsn_ryzen/tokenizer_16k.json'
tokenizer = Tokenizer.from_file(tok_path)

prompts = [
    ('Subject-Verb Agreement', 'The astronomer who observed the distant swirling galaxies'),
    ('Math Reasoning', 'Question: If 5 pens cost 5, how much do 8 pens cost?\nAnswer: Let\'s solve this:'),
    ('Code Syntax', 'def calculate_factorial(n):\n    """Return the factorial of n."""\n'),
    ('Chain-of-Thought', '<think>\nWe need to determine if 9.11 is larger or smaller than 9.9.\n'),
    ('Emergent Syllogism', 'Premise 1: All humans are mortal.\nPremise 2: Socrates is a human.\nConclusion:')
]

for label, p_text in prompts:
    print(f'\n--- [{label}] ---')
    print(f'Prompt: {p_text.strip()}')
    for temp in [0.2, 0.7]:
        input_ids = tokenizer.encode(p_text).ids
        input_tensor = torch.tensor([input_ids], dtype=torch.long)
        torch.manual_seed(42)
        with torch.no_grad():
            for _ in range(50):
                curr = input_tensor if input_tensor.size(1) <= 512 else input_tensor[:, -512:]
                logits = model(curr)
                next_logits = logits[0, -1, :] / temp
                probs = F.softmax(next_logits, dim=-1)
                next_tok = torch.multinomial(probs, num_samples=1)
                input_tensor = torch.cat([input_tensor, next_tok.unsqueeze(0)], dim=1)
        gen_tokens = input_tensor[0, len(input_ids):].tolist()
        gen_text = tokenizer.decode(gen_tokens)
        print(f'[T={temp}]: {repr(gen_text)}')

