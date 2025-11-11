import numpy as np, matplotlib.pyplot as plt
from pathlib import Path

def set_seed(seed):
    import random, torch
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def bar_plot(series, title, out_png):
    counts = series.value_counts().sort_index()
    plt.figure(); counts.plot(kind="bar")
    plt.title(title); plt.xlabel("KL grade"); plt.ylabel("Count")
    plt.tight_layout(); Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png); plt.close()
