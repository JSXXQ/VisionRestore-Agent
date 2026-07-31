from pathlib import Path
import numpy as np
from PIL import Image

out = Path("data/cache/test_images")
out.mkdir(parents=True, exist_ok=True)
h, w = 256, 384
x = np.linspace(0, 1, w)
y = np.linspace(0, 1, h)[:, None]
base = np.dstack([(x * 180 + 30)[None, :].repeat(h, 0), (y * 160 + 40).repeat(w, 1), np.full((h, w), 90)])
imgs = {
    "normal.png": base,
    "low_light.png": base * 0.22,
    "noisy_low_light.png": np.clip(base * 0.18 + np.random.default_rng(1).normal(0, 18, base.shape), 0, 255),
    "color_cast_low_light.png": np.clip(base * [0.35, 0.22, 0.18], 0, 255),
    "local_overexposed.png": base * 0.25,
    "high_resolution.png": np.tile(base * 0.2, (6, 6, 1)),
}
imgs["local_overexposed.png"][70:130, 120:190] = 255
for name, arr in imgs.items():
    Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)).save(out / name)
print(out)
