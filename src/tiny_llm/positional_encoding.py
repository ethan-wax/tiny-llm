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

    def __call__(
        self, x: mx.array, offset: list[slice] | slice | None = None
    ) -> mx.array:
        N, L, H, D = x.shape
        if not offset:
            offset = slice(0, L)
        x = x.reshape([N, L, H, D // 2, 2])
        cos, sin = (
            self.cos_freqs[None, offset, None, :],
            self.sin_freqs[None, offset, None, :],
        )
        real = x[..., 0] * cos - x[..., 1] * sin
        imag = x[..., 0] * sin + x[..., 1] * cos
        out = mx.stack([real, imag], axis=-1)
        return out.reshape([N, L, H, D]).astype(x.dtype)
