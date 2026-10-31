"""
The attention forward ops' segment mask (indrajala-ml's segments workplan, stage 5, D4, D9): with
`segments`, a key in another segment than its query, or a pad's (segment 0), is masked to -inf
before the softmax, but never the diagonal; checked by bits against the crate's own composition
(test_attention_ops.py's), the mask applied to the scores. Without `segments`, and with one segment
a row, the bits are as before.
"""

import math

import numpy as np
import pytest
from array_helpers import bits, to_array, to_numpy
from test_attention_ops import NAMES, _case, _head, _masked, _parameters, _side_by_side

from indrajala_math_rust import affine_forward_batch, array_softmax, attention_forward, attention_forward_batch

# (examples, tokens, features, heads, key_size)
SHAPES = [(1, 4, 3, 1, 3), (2, 6, 4, 2, 2), (3, 8, 6, 3, 2), (2, 64, 8, 4, 2)]


def _segments(examples: int, tokens: int, seed: int) -> np.ndarray:
    # each row: a few pads first, then paragraphs of 1 to 5 tokens
    rng = np.random.default_rng(seed)
    rows = []
    for n in range(examples):
        pads = int(rng.integers(0, max(1, tokens // 4))) if n % 2 else 0
        row, segment = [0.0] * pads, 1
        while len(row) < tokens:
            row += [float(segment)] * int(rng.integers(1, 6))
            segment += 1
        rows.append(row[:tokens])
    return np.array(rows)


def _blocked(segments: np.ndarray) -> np.ndarray:
    blocked = (segments[:, None] != segments[None, :]) | (segments[None, :] == 0.0)
    np.fill_diagonal(blocked, False)
    return blocked


def _composed(c: dict[str, np.ndarray], tokens: int, heads: int, causal: bool, segments: np.ndarray) -> np.ndarray:
    d, width = c["bo"].shape[0], c["bq"].shape[0]
    d_k = width // heads
    rows = to_array(c["x"].reshape(-1, d))
    q, k = (to_numpy(affine_forward_batch(to_array(c["w" + n]), rows, to_array(c["b" + n]))) for n in "qk")
    p: list[list[np.ndarray]] = []
    for n in range(c["x"].shape[0]):
        p.append([])
        for i in range(heads):
            q_ni, k_ni = (to_array(_head(m, n, i, tokens, d_k)) for m in (q, k))
            scores = (q_ni @ k_ni.T) / math.sqrt(d_k)
            if causal:
                scores = _masked(scores, tokens)
            scores = to_array(np.where(_blocked(segments[n]), -np.inf, to_numpy(scores)))
            p[n].append(to_numpy(array_softmax(scores)))
    return _side_by_side(p)


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("shape", SHAPES, ids=["x".join(map(str, s)) for s in SHAPES])
def test_the_weights_are_the_composition_with_the_segment_mask(shape: tuple[int, ...], causal: bool):
    examples, tokens, features, heads, key_size = shape
    c = _case(examples, tokens, features, heads, key_size)
    segments = _segments(examples, tokens, sum(shape))
    result = attention_forward_batch(
        to_array(c["x"]), *_parameters(c), heads=heads, causal=causal, segments=to_array(segments)
    )
    assert bits(to_numpy(result[4])) == bits(_composed(c, tokens, heads, causal, segments))


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
        assert [bits(to_numpy(a)) for a in plain[:6]] == [bits(to_numpy(a)) for a in segmented[:6]]


def test_a_masked_weight_is_exactly_zero_and_a_pad_attends_to_itself():
    c = _case(1, 6, 4, 2, 2)
    segments = np.array([[0.0, 0.0, 1.0, 1.0, 2.0, 2.0]])
    p = to_numpy(attention_forward_batch(to_array(c["x"]), *_parameters(c), heads=2, segments=to_array(segments))[4])
    head = p[:, :6]
    blocked = _blocked(segments[0])
    assert np.all(head[blocked] == 0.0)
    assert head[0].tolist() == [1.0, 0.0, 0.0, 0.0, 0.0, 0.0] and head[1, 1] == 1.0


def test_one_examples_segments_are_its_batch_of_ones():
    c = _case(1, 6, 4, 2, 2)
    segments = np.array([1.0, 1.0, 2.0, 2.0, 2.0, 3.0])
    one = attention_forward(to_array(c["x"][0]), *_parameters(c), heads=2, causal=True, segments=to_array(segments))
    batch = attention_forward_batch(
        to_array(c["x"]), *_parameters(c), heads=2, causal=True, segments=to_array(segments[None])
    )
    assert bits(to_numpy(one[4])) == bits(to_numpy(batch[4]))


def test_segments_of_the_wrong_length_are_refused():
    c = _case(2, 4, 3, 1, 3)
    with pytest.raises(ValueError, match="a segment id per token"):
        attention_forward_batch(to_array(c["x"]), *_parameters(c), heads=1, segments=to_array(np.ones((2, 3))))


assert NAMES  # the parameters' order is test_attention_ops.py's
