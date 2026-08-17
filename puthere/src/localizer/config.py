from dataclasses import dataclass, asdict


@dataclass
class LocalizerConfig:
    encoder_stride: int = 4
    corr_channels: int = 128
    context_channels: int = 64
    context_dilations: tuple = (1, 2, 4, 8, 16, 32, 64, 1, 1)

    heatmap_sigma_cells: float = 2.0

    lambda_offset: float = 1.0
    lambda_hard_negative: float = 0.5
    hard_negative_radius_cells: int = 24
    focal_alpha: float = 2.0
    focal_beta: float = 4.0

    peak_tie_ratio: float = 0.98
    nms_kernel: int = 3

    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 8
    max_steps: int = 40000

    train_seed_lo: int = 0
    train_seed_hi: int = 100_000
    val_seed_lo: int = 100_000
    val_seed_hi: int = 100_500
    test_seed_lo: int = 200_000
    test_seed_hi: int = 200_500
    crops_per_canvas: int = 100

    def as_dict(self) -> dict:
        return asdict(self)
