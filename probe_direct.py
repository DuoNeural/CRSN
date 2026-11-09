import os
import sys
import math
import torch
import torch.nn.functional as F

sys.path.insert(0, "/home/duoneural/crsn_ryzen")
from train_crsn import CRSNLanguageModel, generate, Tokenizer

os.environ["OMP_NUM_THREADS"] = "4"
torch.set_num_threads(4)

ckpt_path = "/home/duoneural/srsn_ryzen/checkpoints_crsn/crsn_best.pt"
print(f"[1/4] Loading {ckpt_path}...")
ckpt = torch.load(ckpt_path, map_location="cpu")
print(f"      Saved Step: {ckpt.get('opt_step')}, Validation Loss: {ckpt.get('loss'):.4f} (Perplexity {math.exp(ckpt.get('loss')):.2f})")

model = CRSNLanguageModel(vocab_size=16384, d_model=384, n_layers=8)
model.load_state_dict(ckpt["model_state"])
model.eval()

# 2. Inspect Clifford Oscillator Parameters
print("\n[2/4] Layer-by-Layer Physical Oscillator Inspection:")
alphas, omegas = [], []
for l_idx, layer in enumerate(model.layers):
    cell = layer.f_module
    alpha = F.softplus(cell.alpha_raw).detach().numpy()
    omega = torch.sqrt(torch.clamp(cell.omega_sq, min=1e-5)).detach().numpy()
    alphas.extend(alpha)
    omegas.extend(omega)
    print(f"  Layer {l_idx:02d}: alpha_mean={alpha.mean():.4f} (min={alpha.min():.4f}, max={alpha.max():.4f}) | omega_mean={omega.mean():.4f}")

import numpy as np
alphas = np.array(alphas)
print(f"  All-layer Alpha Positivity: min={alphas.min():.4f} (Strictly > 0: {np.all(alphas > 0)})")

# 3. Load Tokenizer
tok_path = "/home/duoneural/srsn_ryzen/tokenizer_16k.json"
tokenizer = Tokenizer.from_file(tok_path)

# 4. Qualitative Sampling
prompts = [
    ("Story", "Once upon a time, Lily found a little bird with a broken wing."),
    ("Math CoT", "User: If Alice has 3 boxes and each box has 4 apples, how many apples does Alice have?\n\nAssistant: Let's think step by step:\n"),
    ("Logic", "User: True or False: A dog is an animal.\n\nAssistant: "),
    ("Knowledge", "User: What happens when water freezes?\n\nAssistant: When water freezes, "),
    ("Code", "def fibonacci(n):\n    # Return the nth fibonacci number\n")
]

print("\n[3/4] Autoregressive Generation Probing (80 tokens):")
for label, p_text in prompts:
    print(f"\n=======================================================")
    print(f"PROMPT [{label}]: {repr(p_text)}")
    for temp in [0.3, 0.7]:
        out = generate(model, tokenizer, p_text, max_new_tokens=80, temperature=temp, top_k=40)
        print(f"\n--- [Temp {temp}] ---\n{out}")

print("\n=======================================================")
print("[4/4] Probe Complete.")
