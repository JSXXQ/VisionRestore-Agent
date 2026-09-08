"""Compress UI screenshots for README use.

Old 'final-*.png' were V2.2 UI (3 models) - remove them.
New 'ui-*.{jpg,png}' are V2.3 UI (24 models) - keep, but compress
ui-system-settings.png which is >1MB.
"""
from PIL import Image
import os

BASE = r"E:\codex_project\vision\VisionRestore-Agent\docs\screenshots"

# 1. Remove the old V2.2 screenshots (replaced by ui-* below)
for old in ("final-model-center.png", "final-system-settings.png", "final-workbench.png"):
    p = os.path.join(BASE, old)
    if os.path.exists(p):
        os.remove(p)
        print(f"removed: {old}")

# 2. Compress ui-system-settings.png -> jpg at quality 85
src = os.path.join(BASE, "ui-system-settings.png")
dst = os.path.join(BASE, "ui-system-settings.jpg")
if os.path.exists(src):
    img = Image.open(src).convert("RGB")
    img.save(dst, "JPEG", quality=85, optimize=True)
    os.remove(src)
    print(f"converted: ui-system-settings.png -> ui-system-settings.jpg  ({os.path.getsize(dst)} bytes)")

# 3. Re-encode jpg files at quality 85 (just in case)
for name in ("ui-workbench.jpg", "ui-model-center.jpg"):
    p = os.path.join(BASE, name)
    if os.path.exists(p):
        img = Image.open(p).convert("RGB")
        img.save(p, "JPEG", quality=85, optimize=True)
        print(f"re-encoded: {name}  ({os.path.getsize(p)} bytes)")

# 4. Show final state
print("\nfinal docs/screenshots/:")
for f in sorted(os.listdir(BASE)):
    p = os.path.join(BASE, f)
    print(f"  {f:40s}  {os.path.getsize(p):>10} bytes")
