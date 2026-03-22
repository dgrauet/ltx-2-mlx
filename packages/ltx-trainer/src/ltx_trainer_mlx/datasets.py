"""Dataset implementations for LTX-2 MLX trainer.

Ported from ltx-trainer (Lightricks). Replaces torch.Tensor with mx.array
and torch.load with mx.load / numpy loading. No PyTorch DataLoader needed;
MLX training loops iterate directly.
"""

from pathlib import Path

import mlx.core as mx
import numpy as np

from ltx_trainer_mlx import logger

# Constants for precomputed data directories
PRECOMPUTED_DIR_NAME = ".precomputed"


class DummyDataset:
    """Produce random latents and prompt embeddings for testing and benchmarking."""

    def __init__(
        self,
        width: int = 1024,
        height: int = 1024,
        num_frames: int = 25,
        fps: int = 24,
        dataset_length: int = 200,
        latent_dim: int = 128,
        latent_spatial_compression_ratio: int = 32,
        latent_temporal_compression_ratio: int = 8,
        prompt_embed_dim: int = 4096,
        audio_embed_dim: int = 2048,
        prompt_sequence_length: int = 256,
    ) -> None:
        if width % 32 != 0:
            raise ValueError(f"Width must be divisible by 32, got {width=}")

        if height % 32 != 0:
            raise ValueError(f"Height must be divisible by 32, got {height=}")

        if num_frames % 8 != 1:
            raise ValueError(f"Number of frames must have a remainder of 1 when divided by 8, got {num_frames=}")

        self.width = width
        self.height = height
        self.num_frames = num_frames
        self.fps = fps
        self.dataset_length = dataset_length
        self.latent_dim = latent_dim
        self.num_latent_frames = (num_frames - 1) // latent_temporal_compression_ratio + 1
        self.latent_height = height // latent_spatial_compression_ratio
        self.latent_width = width // latent_spatial_compression_ratio
        self.latent_sequence_length = self.num_latent_frames * self.latent_height * self.latent_width
        self.prompt_embed_dim = prompt_embed_dim
        self.audio_embed_dim = audio_embed_dim
        self.prompt_sequence_length = prompt_sequence_length

    def __len__(self) -> int:
        return self.dataset_length

    def __getitem__(self, idx: int) -> dict[str, dict[str, mx.array]]:
        """Return a sample of random latents and text embeddings.

        Args:
            idx: Sample index (unused, each call produces fresh random data).

        Returns:
            Dict with 'latent_conditions' and 'text_conditions' sub-dicts
            containing mx.array tensors.
        """
        return {
            "latent_conditions": {
                "latents": mx.random.normal(
                    shape=(
                        self.latent_dim,
                        self.num_latent_frames,
                        self.latent_height,
                        self.latent_width,
                    )
                ),
                "num_frames": self.num_latent_frames,
                "height": self.latent_height,
                "width": self.latent_width,
                "fps": self.fps,
            },
            "text_conditions": {
                "video_prompt_embeds": mx.random.normal(
                    shape=(
                        self.prompt_sequence_length,
                        self.prompt_embed_dim,
                    )
                ),
                "audio_prompt_embeds": mx.random.normal(
                    shape=(
                        self.prompt_sequence_length,
                        self.audio_embed_dim,
                    )
                ),
                "prompt_attention_mask": mx.ones(
                    (self.prompt_sequence_length,),
                    dtype=mx.bool_,
                ),
            },
        }


class PrecomputedDataset:
    """Load precomputed latents and conditions from disk as mx.array.

    Supports safetensors (.safetensors) and numpy (.npz/.npy) files.
    """

    def __init__(self, data_root: str, data_sources: dict[str, str] | list[str] | None = None) -> None:
        """Initialize the dataset.

        Args:
            data_root: Root directory containing preprocessed data.
            data_sources: Either:
              - Dict mapping directory names to output keys
              - List of directory names (keys will equal values)
              - None (defaults to ["latents", "conditions"])

        Note:
            Latents are always returned in non-patchified format [C, F, H, W].
            Legacy patchified format [seq_len, C] is automatically converted.
        """
        super().__init__()

        self.data_root = self._setup_data_root(data_root)
        self.data_sources = self._normalize_data_sources(data_sources)
        self.source_paths = self._setup_source_paths()
        self.sample_files = self._discover_samples()
        self._validate_setup()

    @staticmethod
    def _setup_data_root(data_root: str) -> Path:
        """Setup and validate the data root directory."""
        data_root_path = Path(data_root).expanduser().resolve()

        if not data_root_path.exists():
            raise FileNotFoundError(f"Data root directory does not exist: {data_root_path}")

        # If the given path is the dataset root, use the precomputed subdirectory
        if (data_root_path / PRECOMPUTED_DIR_NAME).exists():
            data_root_path = data_root_path / PRECOMPUTED_DIR_NAME

        return data_root_path

    @staticmethod
    def _normalize_data_sources(data_sources: dict[str, str] | list[str] | None) -> dict[str, str]:
        """Normalize data_sources input to a consistent dict format."""
        if data_sources is None:
            return {"latents": "latent_conditions", "conditions": "text_conditions"}
        elif isinstance(data_sources, list):
            return {source: source for source in data_sources}
        elif isinstance(data_sources, dict):
            return data_sources.copy()
        else:
            raise TypeError(f"data_sources must be dict, list, or None, got {type(data_sources)}")

    def _setup_source_paths(self) -> dict[str, Path]:
        """Map data source names to their actual directory paths."""
        source_paths = {}

        for dir_name in self.data_sources:
            source_path = self.data_root / dir_name
            source_paths[dir_name] = source_path

            if not source_path.exists():
                raise FileNotFoundError(f"Required {dir_name} directory does not exist: {source_path}")

        return source_paths

    def _discover_samples(self) -> dict[str, list[Path]]:
        """Discover all valid sample files across all data sources."""
        data_key = "latents" if "latents" in self.data_sources else next(iter(self.data_sources.keys()))
        data_path = self.source_paths[data_key]

        # Support safetensors and numpy files
        data_files: list[Path] = []
        for ext in ("*.safetensors", "*.npz", "*.npy"):
            data_files.extend(data_path.glob(f"**/{ext}"))

        if not data_files:
            raise ValueError(f"No data files found in {data_path}")

        sample_files: dict[str, list[Path]] = {output_key: [] for output_key in self.data_sources.values()}

        for data_file in data_files:
            rel_path = data_file.relative_to(data_path)

            if self._all_source_files_exist(data_file, rel_path):
                self._fill_sample_data_files(data_file, rel_path, sample_files)

        return sample_files

    def _all_source_files_exist(self, data_file: Path, rel_path: Path) -> bool:
        """Check if corresponding files exist in all data sources."""
        for dir_name in self.data_sources:
            expected_path = self._get_expected_file_path(dir_name, data_file, rel_path)
            if not expected_path.exists():
                logger.warning(
                    f"No matching {dir_name} file found for: {data_file.name} (expected in: {expected_path})"
                )
                return False

        return True

    def _get_expected_file_path(self, dir_name: str, data_file: Path, rel_path: Path) -> Path:
        """Get the expected file path for a given data source."""
        source_path = self.source_paths[dir_name]

        # For conditions, handle legacy naming where latent_X maps to condition_X
        if dir_name == "conditions" and data_file.name.startswith("latent_"):
            stem_suffix = data_file.stem[7:]  # strip "latent_" prefix
            return source_path / f"condition_{stem_suffix}{data_file.suffix}"

        return source_path / rel_path

    def _fill_sample_data_files(self, data_file: Path, rel_path: Path, sample_files: dict[str, list[Path]]) -> None:
        """Add a valid sample to the sample_files tracking."""
        for dir_name, output_key in self.data_sources.items():
            expected_path = self._get_expected_file_path(dir_name, data_file, rel_path)
            sample_files[output_key].append(expected_path.relative_to(self.source_paths[dir_name]))

    def _validate_setup(self) -> None:
        """Validate that the dataset setup is correct."""
        if not self.sample_files:
            raise ValueError("No valid samples found - all data sources must have matching files")

        sample_counts = {key: len(files) for key, files in self.sample_files.items()}
        if len(set(sample_counts.values())) > 1:
            raise ValueError(f"Mismatched sample counts across sources: {sample_counts}")

    def __len__(self) -> int:
        first_key = next(iter(self.sample_files.keys()))
        return len(self.sample_files[first_key])

    def __getitem__(self, index: int) -> dict[str, mx.array]:
        """Load a sample from disk as mx.array.

        Args:
            index: Sample index.

        Returns:
            Dict mapping output keys to loaded data (mx.array or nested dicts).
        """
        result: dict[str, mx.array | dict | int] = {}

        for dir_name, output_key in self.data_sources.items():
            source_path = self.source_paths[dir_name]
            file_rel_path = self.sample_files[output_key][index]
            file_path = source_path / file_rel_path

            try:
                data = self._load_file(file_path)

                # Normalize video latent format if this is a latent source
                if "latent" in dir_name.lower():
                    data = self._normalize_video_latents(data)

                result[output_key] = data
            except Exception as e:
                raise RuntimeError(f"Failed to load {output_key} from {file_path}: {e}") from e

        result["idx"] = index
        return result

    @staticmethod
    def _load_file(file_path: Path) -> dict[str, mx.array]:
        """Load data from a file, returning a dict of mx.array.

        Supports .safetensors, .npz, and .npy files.

        Args:
            file_path: Path to the data file.

        Returns:
            Dict mapping tensor names to mx.array values.
        """
        suffix = file_path.suffix.lower()

        if suffix == ".safetensors":
            return dict(mx.load(str(file_path)))

        elif suffix == ".npz":
            np_data = np.load(str(file_path))
            return {k: mx.array(np_data[k]) for k in np_data.files}

        elif suffix == ".npy":
            np_data = np.load(str(file_path))
            return {"data": mx.array(np_data)}

        else:
            raise ValueError(f"Unsupported file format: {suffix}. Use .safetensors or .npz/.npy.")

    @staticmethod
    def _normalize_video_latents(data: dict) -> dict:
        """Normalize video latents to non-patchified format [C, F, H, W].

        Handles backward compatibility with legacy datasets where latents
        are stored in patchified format [seq_len, C].

        Args:
            data: Dict containing at least a 'latents' key.

        Returns:
            Dict with latents guaranteed to be in [C, F, H, W] format.
        """
        if "latents" not in data:
            return data

        latents = data["latents"]

        # Check if latents are in legacy patchified format [seq_len, C]
        if latents.ndim == 2:
            num_frames = int(data["num_frames"])
            height = int(data["height"])
            width = int(data["width"])

            # Unpatchify: [seq_len, C] -> [C, F, H, W]
            # seq_len = F * H * W, so reshape then transpose
            latents = mx.reshape(latents, (num_frames, height, width, -1))
            latents = mx.transpose(latents, (3, 0, 1, 2))

            data = dict(data)
            data["latents"] = latents

        return data
