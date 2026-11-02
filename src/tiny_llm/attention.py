import mlx.core as mx
from .basics import softmax, linear


def scaled_dot_product_attention_simple(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float | None = None,
    mask: mx.array | None = None,
) -> mx.array:
    if scale is None:
        scale = 1 / mx.sqrt(key.shape[-1])

    kq = (query @ key.swapaxes(-1, -2)) * scale
    if mask is not None:
        kq = kq + mask

    attention = softmax(kq, -1) @ value
    return attention


class SimpleMultiHeadAttention:
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        wq: mx.array,
        wk: mx.array,
        wv: mx.array,
        wo: mx.array,
    ):
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.wq = wq
        self.wk = wk
        self.wv = wv
        self.wo = wo

    def __call__(
        self,
        query: mx.array,
        key: mx.array,
        value: mx.array,
        mask: mx.array | None = None,
    ) -> mx.array:
        batch_len, seq_len = query.shape[:2]
        q = (
            linear(query, self.wq)
            .reshape(batch_len, seq_len, self.num_heads, -1)
            .swapaxes(1, 2)
        )
        k = (
            linear(key, self.wk)
            .reshape(batch_len, seq_len, self.num_heads, -1)
            .swapaxes(1, 2)
        )
        v = (
            linear(value, self.wv)
            .reshape(batch_len, seq_len, self.num_heads, -1)
            .swapaxes(1, 2)
        )

        attention = (
            scaled_dot_product_attention_simple(q, k, v, mask=mask)
            .swapaxes(1, 2)
            .reshape(batch_len, seq_len, -1)
        )
        return attention @ self.wo.T


def causal_mask(L: int, S: int, dtype: mx.Dtype) -> mx.array:
    pass


def scaled_dot_product_attention_grouped(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float | None = None,
    mask: mx.array | str | None = None,
) -> mx.array:
    pass


def paged_attention(
    query: mx.array,
    key_pages: mx.array,
    value_pages: mx.array,
    block_table: mx.array,
    context_lens: mx.array,
    page_size: int,
    scale: float | None = None,
    mask: mx.array | str | None = None,
) -> mx.array:
    """Attend to paged K/V storage without reconstructing a dense cache.

    Week 3 Day 4 owns the correctness-first decode and long-prefill paths for
    both float32 and BF16. Day 5 optimizes the same public boundary.
    """
    pass
