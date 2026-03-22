"""Audio processing — STFT and mel filterbank.

Ported from ltx-core audio processing utilities.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np


class AudioProcessor:
    """STFT-based audio processor with mel filterbank.

    Converts raw waveforms to mel spectrograms and back.

    Args:
        sample_rate: Audio sample rate in Hz.
        n_fft: FFT window size.
        hop_length: Hop length for STFT.
        n_mels: Number of mel bands.
        f_min: Minimum frequency for mel filterbank.
        f_max: Maximum frequency for mel filterbank.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 512,
        hop_length: int = 160,
        n_mels: int = 64,
        f_min: float = 0.0,
        f_max: float = 8000.0,
    ):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels

        # Build mel filterbank (numpy for initialization)
        self.mel_basis = mx.array(self._build_mel_filterbank(sample_rate, n_fft, n_mels, f_min, f_max))

        # STFT window
        self.window = mx.array(np.hanning(n_fft + 1)[:-1].astype(np.float32))

    @staticmethod
    def _build_mel_filterbank(sr: int, n_fft: int, n_mels: int, f_min: float, f_max: float) -> np.ndarray:
        """Build a mel filterbank matrix.

        Returns:
            Array of shape (n_mels, n_fft // 2 + 1).
        """

        def hz_to_mel(f: float) -> float:
            return 2595.0 * np.log10(1.0 + f / 700.0)

        def mel_to_hz(m: float) -> float:
            return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

        n_freqs = n_fft // 2 + 1
        mel_min = hz_to_mel(f_min)
        mel_max = hz_to_mel(f_max)
        mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
        hz_points = np.array([mel_to_hz(m) for m in mel_points])
        bin_points = np.floor((n_fft + 1) * hz_points / sr).astype(int)

        filterbank = np.zeros((n_mels, n_freqs), dtype=np.float32)
        for i in range(n_mels):
            left = bin_points[i]
            center = bin_points[i + 1]
            right = bin_points[i + 2]

            for j in range(left, center):
                if center > left:
                    filterbank[i, j] = (j - left) / (center - left)
            for j in range(center, right):
                if right > center:
                    filterbank[i, j] = (right - j) / (right - center)

        return filterbank

    def waveform_to_mel(self, waveform: mx.array) -> mx.array:
        """Convert waveform to mel spectrogram.

        Args:
            waveform: (B, T) or (B, C, T) waveform.

        Returns:
            Mel spectrogram of shape (B, C, T', n_mels) or (B, T', n_mels).
        """
        squeeze_channel = False
        if waveform.ndim == 2:
            waveform = waveform[:, None, :]
            squeeze_channel = True

        B, C, T = waveform.shape
        mels = []

        for b in range(B):
            channel_mels = []
            for c in range(C):
                signal = waveform[b, c]

                # Frame the signal
                num_frames = (T - self.n_fft) // self.hop_length + 1
                indices = mx.arange(self.n_fft)[None, :] + mx.arange(num_frames)[:, None] * self.hop_length
                frames = signal[indices] * self.window

                # FFT
                spec = mx.fft.rfft(frames)
                mag = mx.abs(spec)

                # Mel filterbank
                mel = mag @ self.mel_basis.T
                mel = mx.log(mx.maximum(mel, 1e-5))
                channel_mels.append(mel)

            mels.append(mx.stack(channel_mels, axis=0))

        result = mx.stack(mels, axis=0)
        if squeeze_channel:
            result = result[:, 0]
        return result
