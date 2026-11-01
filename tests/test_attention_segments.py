"""
The attention ops' segment mask (indrajala-ml's segments workplan, D4, D9; its context-length workplan,
D3): with `segments`, a key in another segment than its query, or a pad's (segment 0), is masked to
-inf, but never the diagonal; a key block no query of an example reaches is skipped for it. Checked
by bits against the crate's own tiled composition (test_attention_ops.py's), forward and backward.
Without `segments`, and with one segment a row, the bits are as before.
"""

import numpy as np
import pytest
from array_helpers import bits, to_array, to_numpy
from test_attention_ops import (
    _backward,
    _case,
    _composed_backward,
    _composed_forward,
    _forward,
    _masking,
    _parameters,
)

from indrajala_math_rust import attention_forward, attention_forward_batch

# (examples, tokens, features, heads, key_size)
SHAPES = [(1, 4, 3, 1, 3), (2, 6, 4, 2, 2), (3, 8, 6, 3, 2), (2, 64, 8, 4, 2), (2, 41, 8, 2, 4)]


def _segments(examples: int, tokens: int, seed: int) -> np.ndarray:
    # each row: a few pads first, then paragraphs of 1 to 20 tokens
    rng = np.random.default_rng(seed)
    rows = []
    for n in range(examples):
        pads = int(rng.integers(0, max(1, tokens // 4))) if n % 2 else 0
        row, segment = [0.0] * pads, 1
        while len(row) < tokens:
            row += [float(segment)] * int(rng.integers(1, 21))
            segment += 1
        rows.append(row[:tokens])
    return np.array(rows)


@pytest.mark.parametrize("block", [16, 3])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("shape", SHAPES, ids=["x".join(map(str, s)) for s in SHAPES])
def test_the_passes_are_the_tiled_composition_with_the_segment_mask(shape: tuple[int, ...], causal: bool, block: int):
    examples, tokens, features, heads, key_size = shape
    c = _case(examples, tokens, features, heads, key_size)
    segments = _segments(examples, tokens, sum(shape))
    f = _forward(c, heads, causal, segments=to_array(segments), block=block)
    composed = _composed_forward(c, tokens, heads, causal, block, masking=_masking(causal, segments))
    for name in ["a", "h", "m", "l"]:
        assert bits(f[name]) == bits(composed[name]), name
    b = _backward(c, f, heads, causal, segments=to_array(segments), block=block)
    composed_b = _composed_backward(c, f, tokens, heads, causal, block, masking=_masking(causal, segments))
    for name in ["dx", "dq", "dk", "dv"]:
        assert bits(b[name]) == bits(composed_b[name]), name


@pytest.mark.parametrize("shape", SHAPES, ids=["x".join(map(str, s)) for s in SHAPES])
def test_skipping_blocks_across_segments_gives_the_bits_of_computing_them(shape: tuple[int, ...]):
    examples, tokens, features, heads, key_size = shape
    c = _case(examples, tokens, features, heads, key_size)
    segments = _segments(examples, tokens, sum(shape) + 1)
    f = _forward(c, heads, True, segments=to_array(segments), block=4)
    every = _composed_forward(c, tokens, heads, True, 4, masking=_masking(True, segments), skip=False)
    for name in ["h", "m", "l"]:
        assert bits(f[name]) == bits(every[name]), name


@pytest.mark.parametrize("shape", SHAPES, ids=["x".join(map(str, s)) for s in SHAPES])
def test_one_segment_a_row_is_the_unmasked_pass_by_bits(shape: tuple[int, ...]):
    examples, tokens, features, heads, key_size = shape
    c = _case(examples, tokens, features, heads, key_size)
    ones = to_array(np.ones((examples, tokens)))
    for causal in (False, True):
        plain = attention_forward_batch(to_array(c["x"]), *_parameters(c), heads=heads, causal=causal)
        segmented = attention_forward_batch(
            to_array(c["x"]), *_parameters(c), heads=heads, causal=causal, segments=ones
        )
        assert [bits(to_numpy(a)) for a in plain[:7]] == [bits(to_numpy(a)) for a in segmented[:7]]


def test_a_pad_attends_to_itself_alone():
    c = _case(1, 6, 4, 2, 2)
    segments = np.array([[0.0, 0.0, 1.0, 1.0, 2.0, 2.0]])
    f = _forward(c, 2, segments=to_array(segments))
    # each pad's one weight is exp(0) / 1: its row of H is its row of V, and the paragraphs' first
    # tokens likewise in a causal pass
    assert bits(f["h"][:2]) == bits(f["v"][:2]) and np.all(f["l"][:2] == 1.0)
    causal = _forward(c, 2, True, segments=to_array(segments))
    assert bits(causal["h"][[2, 4]]) == bits(causal["v"][[2, 4]])


def test_one_examples_segments_are_its_batch_of_ones():
    c = _case(1, 6, 4, 2, 2)
    segments = np.array([1.0, 1.0, 2.0, 2.0, 2.0, 3.0])
    one = attention_forward(to_array(c["x"][0]), *_parameters(c), heads=2, causal=True, segments=to_array(segments))
    batch = attention_forward_batch(
        to_array(c["x"]), *_parameters(c), heads=2, causal=True, segments=to_array(segments[None])
    )
    assert [bits(to_numpy(a)) for a in one[1:7]] == [bits(to_numpy(a)) for a in batch[1:7]]


def test_segments_of_the_wrong_length_are_refused():
    c = _case(2, 4, 3, 1, 3)
    with pytest.raises(ValueError, match="a segment id per token"):
        attention_forward_batch(to_array(c["x"]), *_parameters(c), heads=1, segments=to_array(np.ones((2, 3))))
    f = _forward(c, 1)
    with pytest.raises(ValueError, match="a segment id per token"):
        _backward(c, f, 1, segments=to_array(np.ones((2, 3))))
