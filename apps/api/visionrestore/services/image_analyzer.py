import numpy as np
from PIL import Image, ImageFilter, ImageStat
from visionrestore.schemas.image import ImageAnalysisResult

class ImageAnalyzer:
    def analyze(self, path: str) -> ImageAnalysisResult:
        with Image.open(path) as im:
            fmt = im.format or "unknown"
            rgb = im.convert("RGB")
        arr = np.asarray(rgb).astype(np.float32)
        luminance = 0.2126 * arr[..., 0] + 0.7152 * arr[..., 1] + 0.0722 * arr[..., 2]
        hist, _ = np.histogram(luminance, bins=64, range=(0, 255))
        p05, p25, p75, p95 = np.percentile(luminance, [5, 25, 75, 95])
        rgb_means = arr.reshape(-1, 3).mean(axis=0)
        color_cast_index = float(np.std(rgb_means) / (np.mean(rgb_means) + 1e-6))
        if color_cast_index < 0.06:
            cast = "none"
        else:
            cast = ["red", "green", "blue"][int(np.argmax(rgb_means))]
        # Simple edge residual noise estimate. It is an estimate, not a diagnostic truth.
        blur = np.asarray(rgb.filter(ImageFilter.GaussianBlur(radius=1))).astype(np.float32)
        noise_estimate = float(np.std(arr - blur))
        gray = rgb.convert("L")
        sharpness = float(ImageStat.Stat(gray.filter(ImageFilter.FIND_EDGES)).stddev[0] ** 2)
        h, w = luminance.shape
        if h >= 32 and w >= 32:
            crop_h = (h // 32) * 32
            crop_w = (w // 32) * 32
            grid = luminance[:crop_h, :crop_w].reshape(crop_h // 32, 32, crop_w // 32, 32).mean(axis=(1, 3))
        else:
            grid = luminance
        local_non_uniformity = float(np.std(grid) / (np.mean(luminance) + 1e-6))
        pixels = int(w * h)
        notes = ["统计估计值，不能完全替代人工主观判断。"]
        if pixels > 4_000_000:
            notes.append("图像分辨率较高，建议启用分块推理。")
        if noise_estimate > 15:
            notes.append("噪声估计偏高，增强时需要控制噪声放大。")
        return ImageAnalysisResult(
            width=w,
            height=h,
            channels=3,
            format=fmt.lower(),
            bit_depth=8,
            mean_luminance=float(np.mean(luminance)),
            median_luminance=float(np.median(luminance)),
            luminance_p05=float(p05),
            luminance_p25=float(p25),
            luminance_p75=float(p75),
            luminance_p95=float(p95),
            grayscale_histogram=[int(x) for x in hist.tolist()],
            dark_pixel_ratio=float(np.mean(luminance < 50)),
            bright_pixel_ratio=float(np.mean(luminance > 200)),
            overexposed_pixel_ratio=float(np.mean(luminance > 245)),
            rgb_means=[float(x) for x in rgb_means.tolist()],
            color_cast_index=color_cast_index,
            color_cast_label=cast,
            dynamic_range=float(p95 - p05),
            rms_contrast=float(np.std(luminance)),
            laplacian_sharpness=sharpness,
            noise_estimate=noise_estimate,
            local_luminance_non_uniformity=local_non_uniformity,
            suggest_tile_inference=pixels > 4_000_000,
            estimated_memory_mb=float(pixels * 3 * 4 * 6 / 1024 / 1024),
            notes=notes,
        )



