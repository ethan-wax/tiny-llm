import mlx.core as mx


class RMSNorm:
    def __init__(self, dim: int, weight: mx.array, eps: float = 1e-5):
        self.dim = dim
        self.weight = weight
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        dtype = x.dtype
        x_float32 = x.astype(mx.float32)
        square = mx.square(x_float32)
        denom = mx.sqrt(mx.mean(square, axis=-1) + self.eps)[..., None]
        y = (x_float32 / denom).astype(dtype)
        return y * self.weight
