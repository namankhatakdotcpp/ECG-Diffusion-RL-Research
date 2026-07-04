"""
Stage 2 Tier 0 -- shared forward-hook mechanisms.

LIFTED (copied, not moved) from the already-verified, already-committed
scripts below -- those originals are untouched and remain the historical
record for their own items:
  - `_register_layer_hooks`: originally
    stage2_tier0_item1_layerwise_magnitude_direction/layerwise_direction_probe.py
    (itself a derived copy of Stage 1 Experiment 3.5's hook).
  - `RawCaptureHook`, `OverrideHook`: originally
    stage2_tier0_item2_localized_gain/item2_gain_sweep.py (Item 2A).
  - `LocalizedGainHook`: originally
    stage2_tier0_item2_localized_gain/item2_localized_gain.py (Item 2A, A1/A2).

Every item from Item 2B onward should import from here rather than
reimplementing its own copy -- the duplication across Item 1 and Item 2A
is exactly the drift risk this module exists to close off.
"""

from __future__ import annotations

import torch


def register_layer_hooks(model) -> tuple[list, dict]:
    """Registers a forward hook on every model.blocks[i] that mean-pools
    the block's per-token output (B, seq_len, model_dim) -> (B, model_dim)
    and stores it in the returned `captured` dict, keyed by block index.
    Unchanged from Item 1's own `_register_layer_hooks`."""
    captured: dict[int, torch.Tensor] = {}
    handles = []

    def _make_hook(layer_idx: int):
        def _hook(module, inp, out):
            captured[layer_idx] = out.detach().mean(dim=1).cpu()
        return _hook

    for i, block in enumerate(model.blocks):
        handles.append(block.register_forward_hook(_make_hook(i)))
    return handles, captured


def register_block0_input_hook(model) -> tuple[object, dict]:
    """Item 3 addition. Captures block 1's (model.blocks[0]'s) TRUE input
    tensor -- `tokens + cond.unsqueeze(1)` at step04_transformer_diffusion.py:271,
    computed inline between `patch_embed` and the block loop, so it is NOT
    the output of any existing hookable module (confirmed by direct source
    read, Item3_PreRegistration.md). Uses `register_forward_pre_hook`,
    which receives the actual positional args a module is about to be
    called with -- `inp` here is `(tokens, cond_film)`, so `inp[0]` is
    exactly the tensor block 1 consumes. Mean-pooled the same way as
    `register_layer_hooks`, for direct comparability with every other
    captured block-boundary tensor. Blocks 2-6's inputs need no equivalent
    hook: `cond_film` is held constant across the block loop (source-
    verified, step04_transformer_diffusion.py:257-282), so block k's
    output IS block k+1's input, bit-identical -- already captured by
    `register_layer_hooks` with zero new inference."""
    captured: dict[str, torch.Tensor] = {}

    def _pre_hook(module, args):
        tokens = args[0]
        captured["block0_input"] = tokens.detach().mean(dim=1).cpu()

    handle = model.blocks[0].register_forward_pre_hook(_pre_hook)
    return handle, captured


def register_attention_input_hooks(model) -> tuple[list, dict]:
    """Item 6 addition. Captures the FULL per-token tensor each block's
    `self.attn` module receives as its query/key/value input -- this is
    `h = modulate(self.norm1(x), shift1, scale1)` (step04_transformer_
    diffusion.py:173-174, `h, _ = self.attn(h, h, h)`), i.e. the adaLN-
    modulated, normalized representation attention actually operates on,
    NOT the raw block input. `TransformerBlock.forward` discards
    `self.attn`'s attention weights (`h, _ = self.attn(h, h, h)`), so
    there is no way to obtain per-head attention weights from the
    model's own forward pass as written -- the caller must replay
    `block.attn(h, h, h, need_weights=True, average_attn_weights=False)`
    a second time using the tensor this hook captures, in eval mode (no
    dropout), to get per-head weights without altering the model or
    duplicating its forward logic. Captures the FULL tensor (not mean-
    pooled), since attention needs the real per-token sequence."""
    captured: dict[int, torch.Tensor] = {}
    handles = []

    def _make_hook(layer_idx: int):
        def _pre_hook(module, args):
            captured[layer_idx] = args[0].detach().clone()
        return _pre_hook

    for i, block in enumerate(model.blocks):
        handles.append(block.attn.register_forward_pre_hook(_make_hook(i)))
    return handles, captured


class RawCaptureHook:
    """Captures a block's raw, full per-token output tensor (1, seq_len, D)
    -- not mean-pooled. Used to obtain H_k^A(i)/H_k^B(i) so a substitution
    override can be constructed without recomputing the block's forward
    pass for every gain value. Unchanged from Item 2A's own implementation."""

    def __init__(self):
        self.tensor: torch.Tensor | None = None

    def __call__(self, module, inp, out):
        self.tensor = out.detach().clone()
        return out


class OverrideHook:
    """Forward hook on a target block. When `.override` is set (a full
    (1, seq_len, D) tensor), replaces the block's output with it for that
    single forward call; when None, passes the real output through
    unmodified. Unchanged from Item 2A's own implementation. Reusable for
    both the localized (single block) and uniform (multiple blocks,
    cumulative) variants -- for the uniform variant, register one
    independent instance per target block."""

    def __init__(self):
        self.override: torch.Tensor | None = None

    def __call__(self, module, inp, out):
        if self.override is not None:
            return self.override
        return out


class CorrectionHook:
    """Given a precomputed, never-modified class-A reference tensor
    (`cached_A`) and a fixed gain, returns `cached_A + gain*(out - cached_A)`
    every time it fires. This is the live-substitution primitive shared by
    both variants: the localized variant (Item 2A) attaches ONE instance to
    the single target block; the uniform variant (Item 2B) attaches FIVE
    independent instances, one per block 1-5, each with its own cached_A
    (that block's own class-A raw output) and its own per-block gain `g_k` --
    cumulative, since block k+1's hook then sees the already-corrected
    trajectory arriving from block k, not the original class-B path (Item 2
    v3 Sec. 3). Added for Item 2B; extends common/ rather than forking a
    local copy, per the standing instruction not to reimplement hooks
    independently per item."""

    def __init__(self, cached_A: torch.Tensor, gain: float):
        self.cached_A = cached_A
        self.gain = gain

    def __call__(self, module, inp, out):
        delta = out - self.cached_A
        return self.cached_A + self.gain * delta


class LocalizedGainHook:
    """Forward hook on a single target block implementing the cached-mode
    (class-A/class-B) substitution directly, as an alternative calling
    convention to RawCaptureHook+OverrideHook -- kept for parity with
    Item 2A's A1/A2 identity-regression test, which was written against
    this exact class. Unchanged from Item 2A's own implementation."""

    def __init__(self, gain: float):
        self.gain = gain
        self.mode: str | None = None  # "A" or "B", set by the caller before each forward()
        self.cached_A: torch.Tensor | None = None

    def __call__(self, module, inp, out):
        if self.mode == "A":
            self.cached_A = out.detach().clone()
            return out
        elif self.mode == "B":
            if self.cached_A is None:
                raise RuntimeError("Class-A pass must run before class-B pass.")
            delta = out - self.cached_A
            corrected = self.cached_A + self.gain * delta
            return corrected
        else:
            raise RuntimeError("LocalizedGainHook.mode not set before forward().")
