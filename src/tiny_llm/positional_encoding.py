import mlx.core as mx


class RoPE:
    def __init__(
        self,
        dims: int,
        seq_len: int,
        base: int = 10000,
        traditional: bool = False,
    ):
        m = dims // 2
        idx = mx.arange(m)
        angular_rate = mx.power(base, -idx / m)
        pos = mx.arange(seq_len)
        angle = pos[:, None] * angular_rate[None, :]
        self.cos_freqs = mx.cos(angle)
        self.sin_freqs = mx.sin(angle)
        self.traditional = traditional

    def __call__(
        self, x: mx.array, offset: list[slice] | slice | None = None
    ) -> mx.array:
        N, L, H, D = x.shape
        if not offset:
            offset = slice(0, L)
        cos, sin = (
            self.cos_freqs[None, offset, None, :],
            self.sin_freqs[None, offset, None, :],
        )
        if self.traditional:
            x = x.reshape(N, L, H, D // 2, 2)
            real = x[..., 0] * cos - x[..., 1] * sin
            imag = x[..., 0] * sin + x[..., 1] * cos
            out = mx.stack([real, imag], axis=-1).reshape(N, L, H, D)
        else:
            x1, x2 = x[..., : D // 2], x[..., D // 2 :]
            real = x1 * cos - x2 * sin
            imag = x1 * sin + x2 * cos
            out = mx.concatenate([real, imag], axis=-1)
        return out.astype(x.dtype)
