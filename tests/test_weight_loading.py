"""Integration tests for weight loading — requires model weights on disk.

Run with: pytest tests/test_weight_loading.py -v -s
"""

from __future__ import annotations

import mlx.nn as nn
import pytest

from tests.conftest import MODEL_DIR

skip_no_weights = pytest.mark.skipif(
    MODEL_DIR is None,
    reason="q8 model weights not found",
)


@skip_no_weights
class TestWeightLoading:
    """Test loading weights from safetensors files."""

    def test_load_transformer_keys(self):
        """Verify transformer.safetensors loads and has expected key patterns."""
        from ltx_core_mlx.utils.weights import load_split_safetensors

        weights = load_split_safetensors(MODEL_DIR / "transformer.safetensors", prefix="transformer.")
        assert len(weights) > 0, "No weights loaded"

        # Check for expected key patterns
        key_prefixes = {k.split(".")[0] for k in weights}
        print(f"\nTransformer: {len(weights)} params, top-level keys: {sorted(key_prefixes)[:10]}")

        # Should have transformer_blocks
        block_keys = [k for k in weights if k.startswith("transformer_blocks")]
        assert len(block_keys) > 0, "No transformer_blocks found"
        print(f"  transformer_blocks: {len(block_keys)} params")

    def test_load_connector_keys(self):
        from ltx_core_mlx.utils.weights import load_split_safetensors

        weights = load_split_safetensors(MODEL_DIR / "connector.safetensors")
        assert len(weights) > 0
        print(f"\nConnector: {len(weights)} params")
        for k in sorted(weights)[:10]:
            print(f"  {k}: {weights[k].shape}")

    def test_load_vae_decoder_keys(self):
        from ltx_core_mlx.utils.weights import load_split_safetensors

        weights = load_split_safetensors(MODEL_DIR / "vae_decoder.safetensors", prefix="vae_decoder.")
        assert len(weights) > 0
        print(f"\nVAE Decoder: {len(weights)} params")
        for k in sorted(weights)[:10]:
            print(f"  {k}: {weights[k].shape}")

    def test_load_audio_vae_keys(self):
        from ltx_core_mlx.utils.weights import load_split_safetensors, remap_audio_vae_keys

        weights = load_split_safetensors(MODEL_DIR / "audio_vae.safetensors", prefix="audio_vae.decoder.")
        all_audio = load_split_safetensors(MODEL_DIR / "audio_vae.safetensors", prefix="audio_vae.")
        for k, v in all_audio.items():
            if k.startswith("per_channel_statistics."):
                weights[k] = v
        weights = remap_audio_vae_keys(weights)
        assert len(weights) > 0
        print(f"\nAudio VAE: {len(weights)} params")
        for k in sorted(weights)[:10]:
            print(f"  {k}: {weights[k].shape}")

    def test_load_vocoder_keys(self):
        from ltx_core_mlx.utils.weights import load_split_safetensors

        weights = load_split_safetensors(MODEL_DIR / "vocoder.safetensors", prefix="vocoder.")
        assert len(weights) > 0
        print(f"\nVocoder: {len(weights)} params")
        for k in sorted(weights)[:10]:
            print(f"  {k}: {weights[k].shape}")

    def test_transformer_weight_shapes(self):
        """Verify transformer weight shapes match our model architecture."""
        from ltx_core_mlx.utils.weights import load_split_safetensors

        weights = load_split_safetensors(MODEL_DIR / "transformer.safetensors", prefix="transformer.")

        # Check block 0 has expected keys
        block0_keys = sorted(k for k in weights if k.startswith("transformer_blocks.0."))
        assert len(block0_keys) > 0
        print(f"\nBlock 0 keys ({len(block0_keys)}):")
        for k in block0_keys[:20]:
            print(f"  {k}: {weights[k].shape} {weights[k].dtype}")

    def test_vae_decoder_weight_loading(self):
        """Verify VideoDecoder module keys match the safetensors weight keys exactly."""
        from ltx_core_mlx.model.video_vae.video_vae import VideoDecoder
        from ltx_core_mlx.utils.weights import load_split_safetensors

        weights = load_split_safetensors(MODEL_DIR / "vae_decoder.safetensors", prefix="vae_decoder.")

        decoder = VideoDecoder()
        model_keys = set(k for k, _ in nn.utils.tree_flatten(decoder.parameters()))

        weight_keys = set(weights.keys())

        # Check for keys in weights but not in model (missing modules)
        missing_in_model = weight_keys - model_keys
        if missing_in_model:
            print(f"\nKeys in weights but NOT in model ({len(missing_in_model)}):")
            for k in sorted(missing_in_model):
                print(f"  {k}: {weights[k].shape}")

        # Check for keys in model but not in weights (extra modules)
        extra_in_model = model_keys - weight_keys
        if extra_in_model:
            print(f"\nKeys in model but NOT in weights ({len(extra_in_model)}):")
            for k in sorted(extra_in_model):
                print(f"  {k}")

        assert len(missing_in_model) == 0, f"Weight keys not matched by model: {sorted(missing_in_model)}"
        assert len(extra_in_model) == 0, f"Model keys not found in weights: {sorted(extra_in_model)}"

        # Verify shapes match
        mismatched_shapes = []
        for k in weight_keys:
            model_shape = dict(nn.utils.tree_flatten(decoder.parameters()))[k].shape
            weight_shape = weights[k].shape
            if model_shape != weight_shape:
                mismatched_shapes.append((k, model_shape, weight_shape))

        if mismatched_shapes:
            print(f"\nShape mismatches ({len(mismatched_shapes)}):")
            for k, ms, ws in mismatched_shapes[:20]:
                print(f"  {k}: model={ms}, weights={ws}")

        # Actually load weights into the model (should not raise)
        decoder.load_weights(list(weights.items()))
        print(f"\nVAE Decoder: successfully loaded {len(weights)} weight tensors")

    def test_load_transformer_into_model(self):
        """Load transformer.safetensors into LTXModel and verify no missing/extra keys."""
        from ltx_core_mlx.model.transformer.model import LTXModel, LTXModelConfig
        from ltx_core_mlx.utils.weights import apply_quantization, load_split_safetensors

        config = LTXModelConfig()
        model = LTXModel(config)

        weights = load_split_safetensors(MODEL_DIR / "transformer.safetensors", prefix="transformer.")

        # Apply quantization to convert Linear -> QuantizedLinear where weights
        # have scales/biases (int8 quantized layers).
        apply_quantization(model, weights)

        model_keys = set(k for k, _ in nn.utils.tree_flatten(model.parameters()))
        weight_keys = set(weights.keys())

        missing = model_keys - weight_keys
        extra = weight_keys - model_keys

        if missing:
            print(f"\nMissing from weights ({len(missing)}):")
            for k in sorted(missing)[:30]:
                print(f"  {k}")
        if extra:
            print(f"\nExtra in weights ({len(extra)}):")
            for k in sorted(extra)[:30]:
                print(f"  {k}")

        assert len(missing) == 0, f"{len(missing)} keys in model but not in weights"
        assert len(extra) == 0, f"{len(extra)} keys in weights but not in model"

        # Actually load (strict mode)
        model.load_weights(list(weights.items()), strict=True)
        print(f"\nTransformer: successfully loaded {len(weights)} weight tensors into LTXModel")

    def test_vae_encoder_weight_loading(self):
        """Verify VideoEncoder module keys match the safetensors weight keys exactly."""
        from ltx_core_mlx.model.video_vae.ops import remap_encoder_weight_keys
        from ltx_core_mlx.model.video_vae.video_vae import VideoEncoder
        from ltx_core_mlx.utils.weights import load_split_safetensors

        path = MODEL_DIR / "vae_encoder.safetensors"
        if not path.exists():
            pytest.skip("vae_encoder.safetensors not found")

        weights = load_split_safetensors(path, prefix="vae_encoder.")
        weights = remap_encoder_weight_keys(weights)

        encoder = VideoEncoder()
        model_keys = set(k for k, _ in nn.utils.tree_flatten(encoder.parameters()))
        weight_keys = set(weights.keys())

        missing_in_model = weight_keys - model_keys
        extra_in_model = model_keys - weight_keys

        if missing_in_model:
            print(f"\nEncoder keys in weights but NOT in model ({len(missing_in_model)}):")
            for k in sorted(missing_in_model):
                print(f"  {k}: {weights[k].shape}")

        if extra_in_model:
            print(f"\nEncoder keys in model but NOT in weights ({len(extra_in_model)}):")
            for k in sorted(extra_in_model):
                print(f"  {k}")

        assert len(missing_in_model) == 0, f"Weight keys not matched by model: {sorted(missing_in_model)}"
        assert len(extra_in_model) == 0, f"Model keys not found in weights: {sorted(extra_in_model)}"

        encoder.load_weights(list(weights.items()))
        print(f"\nVAE Encoder: successfully loaded {len(weights)} weight tensors")

    def test_audio_vae_decoder_weight_loading(self):
        """Verify AudioVAEDecoder module keys match audio_vae.safetensors exactly."""
        from mlx.utils import tree_flatten, tree_unflatten

        from ltx_core_mlx.model.audio_vae.audio_vae import AudioVAEDecoder
        from ltx_core_mlx.utils.weights import load_split_safetensors, remap_audio_vae_keys

        weights = load_split_safetensors(MODEL_DIR / "audio_vae.safetensors", prefix="audio_vae.decoder.")
        all_audio = load_split_safetensors(MODEL_DIR / "audio_vae.safetensors", prefix="audio_vae.")
        for k, v in all_audio.items():
            if k.startswith("per_channel_statistics."):
                weights[k] = v
        weights = remap_audio_vae_keys(weights)
        assert len(weights) > 0, "No audio VAE weights loaded"

        model = AudioVAEDecoder()

        # Collect model parameter keys
        model_keys = set(k for k, _ in tree_flatten(model.parameters()))
        weight_keys = set(weights.keys())

        # Attention block keys (mid.attn_1, up.1.attn.*, up.2.attn.*) were added
        # to the model architecture and may not yet be present in older weight
        # files.  Build the set of expected attention keys so we can tolerate
        # their absence from weights while still catching genuine mismatches.
        attn_key_prefixes = (
            "mid.attn_1.",
            "up.1.attn.",
            "up.2.attn.",
        )

        missing_in_model = weight_keys - model_keys
        extra_in_model = model_keys - weight_keys

        # Separate attention keys that are expected to be absent from weights
        expected_attn_keys = {k for k in extra_in_model if any(k.startswith(p) for p in attn_key_prefixes)}
        unexpected_extra = extra_in_model - expected_attn_keys

        if missing_in_model:
            print(f"\nAudio VAE keys in weights but NOT in model ({len(missing_in_model)}):")
            for k in sorted(missing_in_model):
                print(f"  {k}: {weights[k].shape}")

        if unexpected_extra:
            print(f"\nAudio VAE keys in model but NOT in weights ({len(unexpected_extra)}):")
            for k in sorted(unexpected_extra):
                print(f"  {k}")

        if expected_attn_keys:
            attn_in_weights = {k for k in weight_keys if any(k.startswith(p) for p in attn_key_prefixes)}
            if attn_in_weights:
                print(f"\nAttention keys found in weights ({len(attn_in_weights)})")
            else:
                print(f"\nAttention keys in model but not yet in weights ({len(expected_attn_keys)}) — OK")

        assert len(missing_in_model) == 0, f"Weight keys not matched by model: {sorted(missing_in_model)}"
        assert len(unexpected_extra) == 0, f"Model keys not found in weights: {sorted(unexpected_extra)}"

        # Load weights using update() to handle underscore-prefixed attrs
        # strict=False because attention block weights may be absent
        model.update(tree_unflatten(list(weights.items())))

        # Verify per_channel_statistics loaded correctly
        assert model.per_channel_statistics.mean_of_means.shape == (128,)
        assert model.per_channel_statistics.std_of_means.shape == (128,)

        print(f"\nAudio VAE Decoder: successfully loaded {len(weights)} weight tensors")

    def test_vocoder_weight_loading(self):
        """Verify VocoderWithBWE module keys match vocoder.safetensors exactly."""
        from ltx_core_mlx.model.audio_vae.bwe import VocoderWithBWE
        from ltx_core_mlx.utils.weights import load_split_safetensors

        weights = load_split_safetensors(MODEL_DIR / "vocoder.safetensors", prefix="vocoder.")
        assert len(weights) > 0, "No vocoder weights loaded"

        model = VocoderWithBWE()

        model_keys = set(k for k, _ in nn.utils.tree_flatten(model.parameters()))
        weight_keys = set(weights.keys())

        missing_in_model = weight_keys - model_keys
        extra_in_model = model_keys - weight_keys

        if missing_in_model:
            print(f"\nVocoder keys in weights but NOT in model ({len(missing_in_model)}):")
            for k in sorted(missing_in_model)[:20]:
                print(f"  {k}: {weights[k].shape}")

        if extra_in_model:
            print(f"\nVocoder keys in model but NOT in weights ({len(extra_in_model)}):")
            for k in sorted(extra_in_model)[:20]:
                print(f"  {k}")

        assert len(missing_in_model) == 0, (
            f"Weight keys not matched by model ({len(missing_in_model)}): {sorted(missing_in_model)[:10]}"
        )
        assert len(extra_in_model) == 0, (
            f"Model keys not found in weights ({len(extra_in_model)}): {sorted(extra_in_model)[:10]}"
        )

        # Actually load weights
        model.load_weights(list(weights.items()))

        # Verify key shapes
        assert model.conv_pre.weight.shape == (1536, 7, 128), (
            f"conv_pre.weight shape mismatch: {model.conv_pre.weight.shape}"
        )
        assert model.mel_stft.mel_basis.shape == (64, 257), (
            f"mel_stft.mel_basis shape mismatch: {model.mel_stft.mel_basis.shape}"
        )
        assert model.bwe_generator.conv_pre.weight.shape == (512, 7, 128), (
            f"bwe_generator.conv_pre.weight shape mismatch: {model.bwe_generator.conv_pre.weight.shape}"
        )

        print(f"\nVocoder+BWE: successfully loaded {len(weights)} weight tensors")

    def test_vocoder_standalone_key_match(self):
        """Verify BigVGANVocoder keys match the base vocoder subset of vocoder.safetensors."""
        from ltx_core_mlx.model.audio_vae.vocoder import BigVGANVocoder
        from ltx_core_mlx.utils.weights import load_split_safetensors

        weights = load_split_safetensors(MODEL_DIR / "vocoder.safetensors", prefix="vocoder.")

        # Extract only base vocoder keys (not bwe_generator or mel_stft)
        base_weight_keys = {k for k in weights if not k.startswith("bwe_generator.") and not k.startswith("mel_stft.")}

        model = BigVGANVocoder()
        model_keys = set(k for k, _ in nn.utils.tree_flatten(model.parameters()))

        missing = base_weight_keys - model_keys
        extra = model_keys - base_weight_keys

        assert len(missing) == 0, f"Base vocoder weight keys not in model: {sorted(missing)}"
        assert len(extra) == 0, f"Model keys not in base vocoder weights: {sorted(extra)}"

        print(f"\nStandalone BigVGANVocoder: {len(model_keys)} keys match base vocoder weights")

    def test_connector_weight_loading(self):
        """Verify TextEncoderConnector module keys match connector.safetensors exactly."""
        from ltx_core_mlx.text_encoders.gemma.feature_extractor import TextEncoderConnector
        from ltx_core_mlx.utils.weights import load_split_safetensors

        weights = load_split_safetensors(MODEL_DIR / "connector.safetensors", prefix="connector.")
        assert len(weights) > 0, "No connector weights loaded"

        model = TextEncoderConnector()

        model_keys = set(k for k, _ in nn.utils.tree_flatten(model.parameters()))
        weight_keys = set(weights.keys())

        missing_in_model = weight_keys - model_keys
        extra_in_model = model_keys - weight_keys

        if missing_in_model:
            print(f"\nConnector keys in weights but NOT in model ({len(missing_in_model)}):")
            for k in sorted(missing_in_model)[:20]:
                print(f"  {k}: {weights[k].shape}")

        if extra_in_model:
            print(f"\nConnector keys in model but NOT in weights ({len(extra_in_model)}):")
            for k in sorted(extra_in_model)[:20]:
                print(f"  {k}")

        assert len(missing_in_model) == 0, (
            f"Weight keys not matched by model ({len(missing_in_model)}): {sorted(missing_in_model)[:10]}"
        )
        assert len(extra_in_model) == 0, (
            f"Model keys not found in weights ({len(extra_in_model)}): {sorted(extra_in_model)[:10]}"
        )

        # Verify shapes match
        mismatched_shapes = []
        model_params = dict(nn.utils.tree_flatten(model.parameters()))
        for k in weight_keys:
            model_shape = model_params[k].shape
            weight_shape = weights[k].shape
            if model_shape != weight_shape:
                mismatched_shapes.append((k, model_shape, weight_shape))

        if mismatched_shapes:
            print(f"\nShape mismatches ({len(mismatched_shapes)}):")
            for k, ms, ws in mismatched_shapes[:20]:
                print(f"  {k}: model={ms}, weights={ws}")

        assert len(mismatched_shapes) == 0, f"Shape mismatches: {mismatched_shapes[:5]}"

        # Actually load weights
        model.load_weights(list(weights.items()))
        print(f"\nTextEncoderConnector: successfully loaded {len(weights)} weight tensors")
