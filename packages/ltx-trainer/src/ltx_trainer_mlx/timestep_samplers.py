"""Timestep samplers for flow matching training.

Ported from ltx-trainer (Lightricks). Replaces torch distributions
with mx.random equivalents.
"""

import mlx.core as mx


class TimestepSampler:
    """Base class for timestep samplers.

    Timestep samplers are used to sample timesteps for diffusion models.
    They should implement both sample() and sample_for() methods.
    """

    def sample(self, batch_size: int, seq_length: int | None = None) -> mx.array:
        """Sample timesteps for a batch.

        Args:
            batch_size: Number of timesteps to sample.
            seq_length: (optional) Length of the sequence being processed.

        Returns:
            Array of shape (batch_size,) containing timesteps.
        """
        raise NotImplementedError

    def sample_for(self, batch: mx.array) -> mx.array:
        """Sample timesteps for a specific batch tensor.

        Args:
            batch: Input array of shape (batch_size, seq_length, ...).

        Returns:
            Array of shape (batch_size,) containing timesteps.
        """
        raise NotImplementedError


class UniformTimestepSampler(TimestepSampler):
    """Samples timesteps uniformly between min_value and max_value (default 0 and 1)."""

    def __init__(self, min_value: float = 0.0, max_value: float = 1.0) -> None:
        self.min_value = min_value
        self.max_value = max_value

    def sample(self, batch_size: int, seq_length: int | None = None) -> mx.array:
        """Sample uniform timesteps.

        Args:
            batch_size: Number of timesteps to sample.
            seq_length: Unused, kept for interface compatibility.

        Returns:
            Array of shape (batch_size,) with values in [min_value, max_value).
        """
        return mx.random.uniform(
            low=self.min_value,
            high=self.max_value,
            shape=(batch_size,),
        )

    def sample_for(self, batch: mx.array) -> mx.array:
        """Sample uniform timesteps matching batch size.

        Args:
            batch: Input array of shape (batch_size, seq_length, ...).

        Returns:
            Array of shape (batch_size,) containing timesteps.

        Raises:
            ValueError: If the input batch does not have 3 dimensions.
        """
        if batch.ndim != 3:
            raise ValueError(f"Batch should have 3 dimensions, got {batch.ndim}")

        return self.sample(batch.shape[0])


class ShiftedLogitNormalTimestepSampler(TimestepSampler):
    """Samples timesteps from a stretched shifted logit-normal distribution.

    The shift is determined by the sequence length. The stretching normalizes
    samples between percentile bounds to ensure the distribution covers [0, 1]
    more evenly. A uniform fallback prevents collapse at high token counts.
    """

    def __init__(self, std: float = 1.0, eps: float = 1e-3, uniform_prob: float = 0.1) -> None:
        self.std = std
        self.eps = eps
        self.uniform_prob = uniform_prob
        # Percentile values for stretching (scaled by std)
        # 99.9th percentile of standard normal ~ 3.0902
        # 0.5th percentile of standard normal ~ -2.5758
        self.normal_999_percentile = 3.0902 * std
        self.normal_005_percentile = -2.5758 * std

    def sample(self, batch_size: int, seq_length: int | None = None) -> mx.array:
        """Sample timesteps from a stretched shifted logit-normal distribution.

        Args:
            batch_size: Number of timesteps to sample.
            seq_length: Length of the sequence being processed, used to determine the shift.

        Returns:
            Array of shape (batch_size,) containing timesteps sampled from a stretched
            shifted logit-normal distribution.

        Raises:
            ValueError: If seq_length is None.
        """
        if seq_length is None:
            raise ValueError("seq_length is required for ShiftedLogitNormalTimestepSampler")

        mu = self._get_shift_for_sequence_length(seq_length)

        # Sample from shifted logit-normal
        normal_samples = mx.random.normal(shape=(batch_size,)) * self.std + mu
        logitnormal_samples = mx.sigmoid(normal_samples)

        # Compute percentile bounds for stretching
        percentile_999 = mx.sigmoid(mx.array(mu + self.normal_999_percentile))
        percentile_005 = mx.sigmoid(mx.array(mu + self.normal_005_percentile))

        # Stretch to [0, 1] range by normalizing between percentiles
        zero_terminal_raw = (logitnormal_samples - percentile_005) / (percentile_999 - percentile_005)

        # Reflect small values around eps for numerical stability
        stretched_logit = mx.where(
            zero_terminal_raw >= self.eps,
            zero_terminal_raw,
            2 * self.eps - zero_terminal_raw,
        )
        stretched_logit = mx.clip(stretched_logit, 0, 1)

        # Mix with uniform samples (uniform_prob of the time)
        uniform = (1 - self.eps) * mx.random.uniform(shape=(batch_size,)) + self.eps
        prob = mx.random.uniform(shape=(batch_size,))

        return mx.where(prob > self.uniform_prob, stretched_logit, uniform)

    def sample_for(self, batch: mx.array) -> mx.array:
        """Sample timesteps for a specific batch tensor.

        Args:
            batch: Input array of shape (batch_size, seq_length, ...).

        Returns:
            Array of shape (batch_size,) containing timesteps sampled from a shifted
            logit-normal distribution.

        Raises:
            ValueError: If the input batch does not have 3 dimensions.
        """
        if batch.ndim != 3:
            raise ValueError(f"Batch should have 3 dimensions, got {batch.ndim}")

        batch_size, seq_length, _ = batch.shape
        return self.sample(batch_size, seq_length)

    @staticmethod
    def _get_shift_for_sequence_length(
        seq_length: int,
        min_tokens: int = 1024,
        max_tokens: int = 4096,
        min_shift: float = 0.95,
        max_shift: float = 2.05,
    ) -> float:
        """Calculate shift value using linear interpolation based on sequence length.

        Args:
            seq_length: Current sequence length.
            min_tokens: Minimum token count for interpolation range.
            max_tokens: Maximum token count for interpolation range.
            min_shift: Shift value at min_tokens.
            max_shift: Shift value at max_tokens.

        Returns:
            Interpolated shift value.
        """
        m = (max_shift - min_shift) / (max_tokens - min_tokens)
        b = min_shift - m * min_tokens
        shift = m * seq_length + b
        return shift


SAMPLERS: dict[str, type[TimestepSampler]] = {
    "uniform": UniformTimestepSampler,
    "shifted_logit_normal": ShiftedLogitNormalTimestepSampler,
}
