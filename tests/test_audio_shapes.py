"""Shape tests for the audio pipeline."""

import mlx.core as mx

from ltx_core_mlx.model.audio_vae.audio_vae import AudioVAEDecoder
from ltx_core_mlx.model.audio_vae.encoder import AudioVAEEncoder
from ltx_core_mlx.model.audio_vae.vocoder import BigVGANVocoder, SnakeBeta


class TestSnakeBeta:
    def test_shape(self):
        act = SnakeBeta(32)
        x = mx.zeros((2, 10, 32))
        out = act(x)
        assert out.shape == (2, 10, 32)

    def test_snakebeta_logscale(self):
        """SnakeBeta with log-scale alpha=0 should give exp(0)=1."""
        snake = SnakeBeta(4)
        # Weights are zeros (log-scale) → exp(0) = 1.0
        x = mx.ones((1, 5, 4))
        result = snake(x)
        # With alpha=1.0, beta=1.0: x + sin^2(x) = 1 + sin^2(1)
        expected_val = 1.0 + mx.sin(mx.array(1.0)).item() ** 2
        assert abs(float(result[0, 0, 0]) - expected_val) < 1e-5


class TestAudioVAEDecoder:
    def test_decode_shape(self):
        decoder = AudioVAEDecoder()
        latent = mx.zeros((1, 8, 10, 16))
        mel = decoder.decode(latent)
        assert mel.shape[0] == 1
        assert mel.shape[1] == 2  # stereo
        # After 2 upsample stages (2x each): freq 16 -> 64
        assert mel.shape[3] == 64  # mel bins


class TestAudioMidBlock:
    def test_mid_block_has_attention(self):
        """MidBlock should have attn_1."""
        from ltx_core_mlx.model.audio_vae.audio_vae import AudioMidBlock

        mid = AudioMidBlock(32)
        assert hasattr(mid, "attn_1")
        x = mx.zeros((1, 4, 4, 32))
        result = mid(x)
        assert result.shape == (1, 4, 4, 32)


class TestAudioVAEEncoder:
    def test_encode_shape(self):
        encoder = AudioVAEEncoder()
        mel = mx.zeros((1, 2, 40, 64))
        latent = encoder.encode(mel)
        assert latent.shape[0] == 1
        assert latent.shape[1] == 8
        assert latent.shape[3] == 16


class TestCausalConv2d:
    def test_causal_upsample_drops_first_row(self):
        """Causal upsample should drop first row."""
        from ltx_core_mlx.model.audio_vae.audio_vae import AudioUpsample

        up = AudioUpsample(4, causal=True)
        x = mx.zeros((1, 4, 4, 4))
        result = up(x)
        # 2x upsample (4->8) minus 1 (drop first row) = 7
        assert result.shape[1] == 7


class TestBigVGANVocoder:
    def test_shape(self):
        vocoder = BigVGANVocoder(
            in_channels=16,
            upsample_initial_channel=32,
            upsample_rates=(2, 2),
            upsample_kernel_sizes=(4, 4),
            resblock_kernel_sizes=(3,),
            resblock_dilation_sizes=((1, 3, 5),),
            out_channels=2,
        )
        mel = mx.zeros((1, 10, 16))
        wav = vocoder(mel)
        assert wav.ndim == 3  # (B, T_audio, out_channels=2)
        assert wav.shape[0] == 1
        assert wav.shape[1] == 10 * 4  # 2 * 2 upsample
        assert wav.shape[2] == 2  # stereo output
