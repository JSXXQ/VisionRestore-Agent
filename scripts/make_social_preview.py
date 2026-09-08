"""Generate GitHub social preview card (1280x640) for VisionRestore Agent.

v2: cleaner layout - title and content on the left 60%, small UI mock at
the bottom-right corner, no overlap. Uses the existing 背景图.png as base.
"""
from PIL import Image, ImageDraw, ImageFilter, ImageFont

OUT = r"E:\codex_project\vision\VisionRestore-Agent\docs\social-preview.png"
BG = r"E:\codex_project\vision\VisionRestore-Agent\背景图.png"
UI_THUMB = r"E:\codex_project\vision\VisionRestore-Agent\docs\screenshots\ui-workbench.jpg"

# Palette derived from the actual UI screenshots (dark + indigo/cyan gradient)
COL_PANEL = (16, 22, 44)
COL_INDIGO = (99, 102, 241)
COL_CYAN = (34, 211, 238)
COL_PURPLE = (168, 85, 247)
COL_TEXT_PRIMARY = (240, 245, 255)
COL_TEXT_MUTED = (170, 185, 210)
COL_BORDER = (60, 75, 110)
COL_BG_FALLBACK = (8, 12, 28)


def load_font(path, size):
    try:
        return ImageFont.truetype(path, size)
    except OSError:
        return ImageFont.load_default()


def chip(draw, x, y, text, font, fg, bg, border, pad_x=14, pad_y=6):
    bbox = draw.textbbox((0, 0), text, font=font)
    w = (bbox[2] - bbox[0]) + pad_x * 2
    h = (bbox[3] - bbox[1]) + pad_y * 2
    draw.rounded_rectangle((x, y, x + w, y + h), radius=h // 2,
                           fill=bg, outline=border, width=1)
    draw.text((x + pad_x, y + pad_y - bbox[1]), text, font=font, fill=fg)
    return w, h


def main():
    W, H = 1280, 640

    # --- Background ---
    try:
        bg = Image.open(BG).convert("RGBA")
        scale = max(W / bg.width, H / bg.height)
        nw, nh = int(bg.width * scale), int(bg.height * scale)
        bg = bg.resize((nw, nh), Image.LANCZOS)
        x0 = (W - nw) // 2
        y0 = (H - nh) // 2
        canvas = Image.new("RGBA", (W, H), COL_BG_FALLBACK + (255,))
        canvas.paste(bg, (x0, y0))
    except FileNotFoundError:
        canvas = Image.new("RGBA", (W, H), COL_BG_FALLBACK + (255,))

    # Vignette
    vignette = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    vd = ImageDraw.Draw(vignette)
    for r in range(0, 500, 10):
        alpha = max(0, int(180 - r * 0.35))
        vd.ellipse((W // 2 - r * 2, H // 2 - r * 2, W // 2 + r * 2, H // 2 + r * 2),
                   outline=(0, 0, 0, alpha), width=10)
    vignette = vignette.filter(ImageFilter.GaussianBlur(40))
    canvas = Image.alpha_composite(canvas, vignette)

    # Dark left gradient for text readability
    left_grad = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    lg = ImageDraw.Draw(left_grad)
    for x in range(W):
        if x < 800:
            t = 1 - x / 800
            a = int(180 * t)
            lg.line([(x, 0), (x, H)], fill=(0, 0, 0, a))
    canvas = Image.alpha_composite(canvas, left_grad)

    draw = ImageDraw.Draw(canvas)

    # --- Fonts ---
    f_title = load_font(r"C:\Windows\Fonts\msyhbd.ttc", 86)
    f_sub_cn = load_font(r"C:\Windows\Fonts\msyh.ttc", 30)
    f_sub_en = load_font(r"C:\Windows\Fonts\arial.ttf", 28)
    f_chip_en = load_font(r"C:\Windows\Fonts\arialbd.ttf", 22)
    f_chip_cn = load_font(r"C:\Windows\Fonts\msyh.ttc", 22)
    f_meta = load_font(r"C:\Windows\Fonts\arial.ttf", 22)
    f_label = load_font(r"C:\Windows\Fonts\msyh.ttc", 18)

    # --- Top badge row ---
    left_x = 80
    top_y = 70
    bw, _ = chip(draw, left_x, top_y, "V2.3  ·  LangGraph Multi-Candidate",
                 f_chip_en, COL_CYAN, (16, 32, 56, 220), COL_CYAN, pad_x=18, pad_y=8)
    cur_x = left_x + bw + 12
    bw2, _ = chip(draw, cur_x, top_y, "Local CUDA  ·  No Cloud Key",
                  f_chip_en, COL_TEXT_PRIMARY, (28, 28, 60, 220), COL_INDIGO,
                  pad_x=18, pad_y=8)
    cur_x += bw2 + 12
    chip(draw, cur_x, top_y, "MIT",
         f_chip_en, COL_TEXT_PRIMARY, (60, 40, 110, 220), COL_PURPLE, pad_x=18, pad_y=8)

    # --- Title ---
    title_y = 170
    draw.text((left_x, title_y), "VisionRestore Agent",
              font=f_title, fill=COL_TEXT_PRIMARY)

    # --- Subtitle (CN + EN on one line) ---
    sub_y = title_y + 120
    draw.text((left_x, sub_y), "多模型协同  ·  低照度图像增强 Agent",
              font=f_sub_cn, fill=COL_TEXT_MUTED)
    draw.text((left_x, sub_y + 44), "Multi-Model Low-Light Image Restoration",
              font=f_sub_en, fill=COL_TEXT_MUTED)

    # --- Description line ---
    desc_y = sub_y + 110
    draw.text((left_x, desc_y),
              "Retinexformer · DarkIR · HVI-CIDNet · FLOL · SCI  +  NAFNet / Real-ESRGAN",
              font=f_meta, fill=COL_TEXT_PRIMARY)

    # --- Tech chips row ---
    chips_y = desc_y + 56
    chip_specs = [
        ("Python 3.12+", f_chip_en, COL_CYAN),
        ("PyTorch 2.5", f_chip_en, COL_CYAN),
        ("FastAPI", f_chip_en, COL_CYAN),
        ("React + Vite", f_chip_en, COL_CYAN),
        ("LangGraph", f_chip_en, COL_INDIGO),
        ("CUDA 12.1", f_chip_en, COL_INDIGO),
        ("SQLite", f_chip_en, COL_INDIGO),
    ]
    cx = left_x
    for text, font, color in chip_specs:
        bw, _ = chip(draw, cx, chips_y, text, font, COL_TEXT_PRIMARY,
                     (24, 28, 50, 220), color, pad_x=18, pad_y=8)
        cx += bw + 10

    # --- Bottom meta ---
    meta_y = H - 60
    draw.text((left_x, meta_y), "github.com/JSXXQ/VisionRestore-Agent",
              font=f_meta, fill=COL_TEXT_MUTED)

    # --- Bottom-right UI thumbnail ---
    thumb_w = 380
    thumb_h = 220
    thumb_x = W - thumb_w - 70
    thumb_y = H - thumb_h - 80

    # Shadow
    shadow = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    sd = ImageDraw.Draw(shadow)
    sd.rounded_rectangle((thumb_x + 6, thumb_y + 10, thumb_x + thumb_w + 6, thumb_y + thumb_h + 10),
                         radius=18, fill=(0, 0, 0, 180))
    shadow = shadow.filter(ImageFilter.GaussianBlur(16))
    canvas = Image.alpha_composite(canvas, shadow)
    draw = ImageDraw.Draw(canvas)

    # Frame
    draw.rounded_rectangle((thumb_x - 2, thumb_y - 2, thumb_x + thumb_w + 2, thumb_y + thumb_h + 2),
                           radius=18, outline=COL_INDIGO, width=2)
    # Thumbnail
    try:
        thumb = Image.open(UI_THUMB).convert("RGB")
        thumb = thumb.resize((thumb_w, thumb_h), Image.LANCZOS)
        # rounded mask
        mask = Image.new("L", (thumb_w, thumb_h), 0)
        md = ImageDraw.Draw(mask)
        md.rounded_rectangle((0, 0, thumb_w, thumb_h), radius=16, fill=255)
        thumb_rgba = thumb.convert("RGBA")
        thumb_rgba.putalpha(mask)
        canvas.paste(thumb_rgba, (thumb_x, thumb_y), thumb_rgba)
        draw = ImageDraw.Draw(canvas)
    except FileNotFoundError:
        # Fallback: dark panel
        draw.rounded_rectangle((thumb_x, thumb_y, thumb_x + thumb_w, thumb_y + thumb_h),
                               radius=16, fill=COL_PANEL + (200,), outline=COL_BORDER, width=2)
        draw.text((thumb_x + 20, thumb_y + 20), "[UI Preview]", font=f_label, fill=COL_TEXT_MUTED)

    # Caption above the thumb
    cap_y = thumb_y - 30
    draw.text((thumb_x, cap_y), "UI Preview  ·  增强工作台", font=f_chip_cn, fill=COL_TEXT_MUTED)

    # --- Save ---
    out = canvas.convert("RGB")
    out.save(OUT, "PNG", optimize=True)
    import os
    print(f"OK -> {OUT}  size={out.size}  bytes={os.path.getsize(OUT)}")


if __name__ == "__main__":
    main()
