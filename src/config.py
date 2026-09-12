"""
config.py
Strongly-typed configuration system and CLI override parser for Edge-AUI Framework.
Loads settings from src/config.yaml with fallback defaults and environment detection.
"""

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

# Single source of truth for MicroTensor dimensionality
NUM_BEHAVIOURAL_FEATURES: int = 9
MICROTENSOR_DIM: int = 18  # 9 features + 9 binary modality masks = 18


@dataclass
class DataConfig:
    raw_dir: str = ".data/raw"
    canonical_dir: str = ".data/canonical"
    interim_dir: str = ".data/interim"
    processed_dir: str = ".data/processed"
    dvc_remote: str = "hfremote"
    hf_repo: str = "T40/edge-aui-framework-data"
    consolidate_parquet: bool = True


@dataclass
class NormalizationConfig:
    mean_velocity_scale: float = 10.0
    max_velocity_scale: float = 10.0
    mean_acceleration_scale: float = 1.0
    hesitation_scale: float = 25.0
    trajectory_scale: float = 2000.0
    scroll_velocity_scale: float = 5.0


@dataclass
class PreprocessingConfig:
    window_size_ms: int = 500
    stride_ms: int = 250
    reference_viewport: Tuple[int, int] = (1920, 1080)
    core_features: List[str] = field(default_factory=lambda: [
        "meanVelocity",
        "maxVelocity",
        "meanAcceleration",
        "hesitationCount",
        "totalTrajectoryLength",
        "dwellTimeMs",
        "trajectoryEntropy"
    ])
    contextual_features: List[str] = field(default_factory=lambda: [
        "scrollDepthPercentage",
        "scrollVelocity"
    ])
    num_features: int = NUM_BEHAVIOURAL_FEATURES
    input_dim: int = MICROTENSOR_DIM  # 9 features * 2 (with binary modality mask [X * M, M])
    min_events_per_window: int = 3
    normalization: NormalizationConfig = field(default_factory=NormalizationConfig)


@dataclass
class TargetGenerationConfig:
    lookahead_horizon_ms: Tuple[int, int] = (500, 1500)
    outcome_taxonomy: Dict[int, str] = field(default_factory=lambda: {
        0: "IDLE_ABANDON",
        1: "CLICK",
        2: "FORM_SUBMIT",
        3: "BACKTRACK",
        4: "RAPID_SCROLL",
        5: "HOVER_DWELL"
    })
    target_intervention_vocabulary: Dict[int, str] = field(default_factory=lambda: {
        0: "simplify_options",
        1: "highlight_primary_action",
        2: "offer_assistance",
        3: "expand_tooltip",
        4: "no_op"
    })


@dataclass
class TrainingConfig:
    batch_size: int = 64
    epochs: int = 5
    learning_rate: float = 0.001
    hidden_dim: int = 64
    num_layers: int = 2
    foundation_classes: int = 6
    target_classes: int = 5
    device: str = "auto"


@dataclass
class AblationConfig:
    strict_freezing_head_lr: float = 0.001
    partial_gru_lr: float = 0.0001
    partial_head_lr: float = 0.001
    full_lr: float = 0.0005
    metrics: List[str] = field(default_factory=lambda: ["hr_1", "hr_3", "mrr"])


@dataclass
class ExportConfig:
    output_dir: str = "models"
    onnx_filename: str = "model.onnx"
    quantized_filename: str = "model_int8.onnx"
    quantization_type: str = "QUInt8"
    opset_version: int = 17
    max_memory_mb: float = 20.0
    target_latency_ms: float = 50.0


@dataclass
class PipelineConfig:
    mode: str = "minimal"
    modes: Dict[str, Any] = field(default_factory=lambda: {
        "minimal": {
            "description": "MVP: AdSERP -> Target UI Testbed",
            "sequence": ["adserp", "target_testbed"]
        },
        "extended": {
            "description": "Extended: CaptchaSolve30k -> VideoCUA -> AdSERP -> Target UI Testbed",
            "sequence": ["captcha_solve_30k", "video_cua", "adserp", "target_testbed"]
        }
    })
    data: DataConfig = field(default_factory=DataConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    target_generation: TargetGenerationConfig = field(default_factory=TargetGenerationConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    ablation: AblationConfig = field(default_factory=AblationConfig)
    export: ExportConfig = field(default_factory=ExportConfig)

    @property
    def input_dim(self) -> int:
        return self.preprocessing.input_dim


def find_config_file(custom_path: Optional[str] = None) -> Optional[Path]:
    """Locate config.yaml in common project locations."""
    if custom_path and os.path.exists(custom_path):
        return Path(custom_path)

    curr = Path(__file__).resolve().parent
    candidates = [
        curr / "config.yaml",
        curr.parent / "src" / "config.yaml",
        curr.parent / "config.yaml",
        Path("src/config.yaml"),
        Path("config.yaml")
    ]
    for c in candidates:
        if c.is_file():
            return c
    return None


def _parse_yaml_fallback(text: str) -> Dict[str, Any]:
    """
    Lightweight fallback parser for nested key-values if PyYAML is not installed.
    Supports basic scalars, lists, and dicts used in config.yaml.
    """
    res: Dict[str, Any] = {}
    current_section = None

    for line in text.splitlines():
        clean = line.strip()
        if not clean or clean.startswith("#"):
            continue
        # Top-level key
        if not line.startswith(" ") and ":" in line:
            parts = clean.split(":", 1)
            k = parts[0].strip()
            v = parts[1].strip()
            if not v or v.startswith("#"):
                current_section = k
                res[k] = {}
            else:
                current_section = None
                # Clean comments
                v_clean = v.split("#")[0].strip()
                res[k] = _cast_value(v_clean)
        elif line.startswith("  ") and not line.startswith("    ") and ":" in line and current_section:
            parts = clean.split(":", 1)
            k = parts[0].strip()
            v = parts[1].split("#")[0].strip()
            if v:
                res[current_section][k] = _cast_value(v)
            else:
                res[current_section][k] = []

    return res


def _cast_value(val_str: str) -> Any:
    val_str = val_str.strip().strip("'\"")
    if val_str.lower() in ("true", "yes"):
        return True
    if val_str.lower() in ("false", "no"):
        return False
    try:
        if "." in val_str:
            return float(val_str)
        return int(val_str)
    except ValueError:
        return val_str


def load_raw_yaml(path: Path) -> Dict[str, Any]:
    """Load dictionary from YAML file with PyYAML or safe fallback."""
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    try:
        import yaml
        return yaml.safe_load(content) or {}
    except ImportError:
        return _parse_yaml_fallback(content)


def load_config(
    mode: Optional[str] = None,
    config_path: Optional[str] = None,
    **overrides: Any
) -> PipelineConfig:
    """
    Load pipeline configuration with hierarchical resolution:
    1. Base dataclass defaults
    2. config.yaml values
    3. Keyword argument overrides (CLI parameters)
    """
    cfg_file = find_config_file(config_path)
    raw_dict: Dict[str, Any] = {}
    if cfg_file:
        raw_dict = load_raw_yaml(cfg_file)

    selected_mode = mode or overrides.get("mode") or raw_dict.get("mode", "minimal")

    # Construct nested dataclasses
    data_dict = raw_dict.get("data", {})
    data_cfg = DataConfig(
        raw_dir=overrides.get("raw_dir", data_dict.get("raw_dir", ".data/raw")),
        canonical_dir=overrides.get("canonical_dir", data_dict.get("canonical_dir", ".data/canonical")),
        interim_dir=overrides.get("interim_dir", data_dict.get("interim_dir", ".data/interim")),
        processed_dir=overrides.get("processed_dir", data_dict.get("processed_dir", ".data/processed")),
        dvc_remote=data_dict.get("dvc_remote", "hfremote"),
        hf_repo=overrides.get("hf_repo", data_dict.get("hf_repo", "T40/edge-aui-framework-data")),
        consolidate_parquet=data_dict.get("consolidate_parquet", True)
    )

    prep_dict = raw_dict.get("preprocessing", {})
    norm_dict = prep_dict.get("normalization", {})
    norm_cfg = NormalizationConfig(
        mean_velocity_scale=norm_dict.get("mean_velocity_scale", 10.0),
        max_velocity_scale=norm_dict.get("max_velocity_scale", 10.0),
        mean_acceleration_scale=norm_dict.get("mean_acceleration_scale", 0.1),
        hesitation_scale=norm_dict.get("hesitation_scale", 10.0),
        trajectory_scale=norm_dict.get("trajectory_scale", 2000.0),
        scroll_velocity_scale=norm_dict.get("scroll_velocity_scale", 5.0)
    )
    prep_cfg = PreprocessingConfig(
        window_size_ms=prep_dict.get("window_size_ms", 500),
        stride_ms=prep_dict.get("stride_ms", 250),
        num_features=prep_dict.get("num_features", NUM_BEHAVIOURAL_FEATURES),
        input_dim=prep_dict.get("input_dim", MICROTENSOR_DIM),
        min_events_per_window=prep_dict.get("min_events_per_window", 3),
        normalization=norm_cfg
    )

    train_dict = raw_dict.get("training", {})
    train_cfg = TrainingConfig(
        batch_size=overrides.get("batch_size", train_dict.get("batch_size", 64)),
        epochs=overrides.get("epochs", train_dict.get("epochs", 5)),
        learning_rate=overrides.get("lr", train_dict.get("learning_rate", 0.001)),
        hidden_dim=overrides.get("hidden_dim", train_dict.get("hidden_dim", 64)),
        num_layers=overrides.get("num_layers", train_dict.get("num_layers", 2)),
        device=overrides.get("device", train_dict.get("device", "auto"))
    )

    export_dict = raw_dict.get("export", {})
    export_cfg = ExportConfig(
        output_dir=overrides.get("output_dir", export_dict.get("output_dir", "models")),
        onnx_filename=export_dict.get("onnx_filename", "model.onnx"),
        quantized_filename=export_dict.get("quantized_filename", "model_int8.onnx"),
        max_memory_mb=export_dict.get("max_memory_mb", 20.0),
        target_latency_ms=export_dict.get("target_latency_ms", 50.0)
    )

    pipeline_cfg = PipelineConfig(
        mode=selected_mode,
        data=data_cfg,
        preprocessing=prep_cfg,
        target_generation=TargetGenerationConfig(),
        training=train_cfg,
        ablation=AblationConfig(),
        export=export_cfg
    )

    return pipeline_cfg


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Edge-AUI Pipeline Configuration Inspector")
    parser.add_argument("--mode", type=str, default="minimal", choices=["minimal", "extended"])
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(mode=args.mode, config_path=args.config)
    print(f"Loaded Edge-AUI Config:")
    print(f" - Mode:             {cfg.mode}")
    print(f" - Sequence:         {cfg.modes.get(cfg.mode, {}).get('sequence')}")
    print(f" - Feature Dim:      {cfg.preprocessing.num_features} (Input with mask: {cfg.input_dim})")
    print(f" - Recurrent Hidden: {cfg.training.hidden_dim}")
    print(f" - Window/Stride:    {cfg.preprocessing.window_size_ms}ms / {cfg.preprocessing.stride_ms}ms")
    print(f" - Data Partitions:  raw={cfg.data.raw_dir}, interim={cfg.data.interim_dir}")
