from pydantic import BaseModel, Field

class FileRecord(BaseModel):
    file_id: str
    relative_path: str
    sha256: str
    mime_type: str
    width: int
    height: int
    size_bytes: int
    uploaded_at: str

class ImageAnalysisResult(BaseModel):
    width: int
    height: int
    channels: int
    format: str
    bit_depth: int
    mean_luminance: float
    median_luminance: float
    luminance_p05: float
    luminance_p25: float
    luminance_p75: float
    luminance_p95: float
    grayscale_histogram: list[int] = Field(default_factory=list)
    dark_pixel_ratio: float
    bright_pixel_ratio: float
    overexposed_pixel_ratio: float
    rgb_means: list[float]
    color_cast_index: float
    color_cast_label: str
    dynamic_range: float
    rms_contrast: float
    laplacian_sharpness: float
    noise_estimate: float
    image_entropy: float = 0
    local_luminance_non_uniformity: float
    total_pixels: int = 0
    suggest_tile_inference: bool
    estimated_memory_mb: float
    notes: list[str] = Field(default_factory=list)
