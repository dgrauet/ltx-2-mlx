# LTX-2 TeaCache Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire TeaCache (timestep-aware residual caching) into LTX-2 MLX as an opt-in inference accelerator for `TwoStagePipeline` stage 1 (non-distilled, 30 Euler steps), and ship a calibration script the user runs on their 32 GB host to produce the LTX-2 polyfit coefficients.

**Architecture:** Two control hooks added to `LTXModel.__call__`: a `tap` callable (observation only, used by calibration) and a `block_stack_override` callable (replaces the block iteration with a cached-residual reconstruction, used by TeaCache skip path). A new `compute_gate_signal` method on `LTXModel` returns block 0's modulated input cheaply (prelude only) so the sampler can decide skip-or-compute before invoking the full forward. The sampler `guided_denoise_loop` owns the `TeaCacheController`, drives the per-step decision, and manages the per-pass residual dict (cond / uncond / ptb / mod). `mlx-arsenal`'s controller gets a one-line type relaxation so its residual storage accepts `Any` (a dict here).

**Tech Stack:** MLX (`mlx>=0.31.0`), `numpy` (polyfit), pytest, ruff. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-04-26-ltx2-teacache-calibration-design.md`.

---

## File Structure

**mlx-arsenal:**
- Modify: `mlx_arsenal/diffusion/teacache.py` — relax `cache_residual` / `previous_residual` type from `mx.array` to `Any`.
- Modify: `tests/test_teacache.py` — one test for non-array cached payloads (dict).

**ltx-2-mlx — `ltx-core-mlx`:**
- Modify: `packages/ltx-core-mlx/src/ltx_core_mlx/model/transformer/transformer.py` — add `BasicAVTransformerBlock.compute_video_normed_sa` helper.
- Modify: `packages/ltx-core-mlx/src/ltx_core_mlx/model/transformer/model.py` — add `LTXModel.compute_gate_signal`, plumb `tap` + `block_stack_override` through `LTXModel.__call__` and `X0Model.__call__`.

**ltx-2-mlx — `ltx-pipelines-mlx`:**
- Modify: `packages/ltx-pipelines-mlx/src/ltx_pipelines_mlx/utils/samplers.py` — add `tap` and `teacache` kwargs to `guided_denoise_loop` and integrate per-pass residual handling.
- Modify: `packages/ltx-pipelines-mlx/src/ltx_pipelines_mlx/ti2vid_two_stages.py` — add `enable_teacache: bool = False` and `teacache_thresh: float | None = None` kwargs to `generate_two_stage`.
- Create: `packages/ltx-pipelines-mlx/scripts/calibrate_teacache.py` — calibration entry point.

**ltx-2-mlx — tests:**
- Modify: `tests/test_model_shapes.py` — tests for new transformer/model hooks (use existing tiny-config pattern).
- Create: `tests/test_teacache_integration.py` — integration tests for the sampler-level TeaCache flow (tiny model, stubbed controller).

---

## Task 1: mlx-arsenal — relax `TeaCacheController` cached-residual type

**Files:**
- Modify: `/Users/dgrauet/Work/mlx-arsenal/mlx_arsenal/diffusion/teacache.py`
- Modify: `/Users/dgrauet/Work/mlx-arsenal/tests/test_teacache.py`

**Why:** LTX-2 caches a per-pass dict (`{"cond": (v_res, a_res), "uncond": (v_res, a_res)}`). The current single-tensor type is too narrow.

- [ ] **Step 1: Add the failing test**

Append this class to `/Users/dgrauet/Work/mlx-arsenal/tests/test_teacache.py` (after `class TestEndToEndCacheReuse`):

```python
class TestArbitraryPayloadCache:
    def test_cache_residual_accepts_dict_of_tuples(self):
        """LTX-2 caches a per-pass dict; controller must accept arbitrary payloads."""
        c = make_controller()
        payload = {
            "cond": (mx.array([1.0]), mx.array([2.0])),
            "uncond": (mx.array([3.0]), mx.array([4.0])),
        }
        c.cache_residual(payload)
        retrieved = c.previous_residual
        assert retrieved is payload  # exact identity, not a copy
        assert mx.allclose(retrieved["cond"][0], mx.array([1.0])).item()
        assert mx.allclose(retrieved["uncond"][1], mx.array([4.0])).item()
```

- [ ] **Step 2: Run the test to verify it currently passes (it should, because Python is dynamic)**

Run: `cd /Users/dgrauet/Work/mlx-arsenal && python3 -m pytest tests/test_teacache.py::TestArbitraryPayloadCache -v`

Expected: PASS (the existing implementation stores whatever is passed). The test guards the contract against future tightening.

- [ ] **Step 3: Loosen the type annotations**

In `/Users/dgrauet/Work/mlx-arsenal/mlx_arsenal/diffusion/teacache.py`:

Replace the `cache_residual` method:

```python
    def cache_residual(self, residual) -> None:
        """Store the residual from the just-computed step for reuse on skip.

        ``residual`` is whatever the caller wants to retrieve later via
        ``previous_residual``. Single-tensor models pass an ``mx.array``;
        multi-stream models (e.g. LTX-2) pass a tuple or dict. The controller
        does not inspect or copy the value.
        """
        self._prev_residual = residual
```

And the `previous_residual` property (replace the existing one):

```python
    @property
    def previous_residual(self):
        """Last cached payload. Raises before the first ``cache_residual`` call."""
        if self._prev_residual is None:
            raise RuntimeError(
                "No residual cached yet — call cache_residual() after a computed step "
                "before reading previous_residual."
            )
        return self._prev_residual
```

Also change the field annotation in `__init__`:

```python
        self._prev_residual = None
```

(was `self._prev_residual: mx.array | None = None`.)

- [ ] **Step 4: Run the full arsenal test suite**

Run: `cd /Users/dgrauet/Work/mlx-arsenal && python3 -m pytest tests/ -q`

Expected: 144 passed (143 existing + 1 new), 0 failed.

- [ ] **Step 5: Run ruff**

Run: `cd /Users/dgrauet/Work/mlx-arsenal && python3 -m ruff check mlx_arsenal/diffusion/teacache.py tests/test_teacache.py`

Expected: `All checks passed!`

- [ ] **Step 6: Update CHANGELOG**

In `/Users/dgrauet/Work/mlx-arsenal/CHANGELOG.md`, under `[Unreleased]` → `Added` (append below the existing TeaCache lines), add a new `Changed` subsection:

```markdown
### Changed
- `TeaCacheController.cache_residual` / `previous_residual` now accept any
  payload (previously typed as `mx.array`). Enables multi-stream / multi-pass
  models (e.g. LTX-2) to cache a tuple or dict of residuals.
```

- [ ] **Step 7: Commit**

```bash
cd /Users/dgrauet/Work/mlx-arsenal
git add mlx_arsenal/diffusion/teacache.py tests/test_teacache.py CHANGELOG.md
git commit -m "$(cat <<'EOF'
feat: relax TeaCacheController residual type to Any

Multi-stream models (LTX-2 with audio+video, optionally multi-pass with
CFG) need to cache a structured payload (tuple/dict) per skip step. The
controller is structure-agnostic; relax annotations and document.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: `BasicAVTransformerBlock.compute_video_normed_sa` — gate-signal helper

**Files:**
- Modify: `/Users/dgrauet/Work/ltx-2-mlx/packages/ltx-core-mlx/src/ltx_core_mlx/model/transformer/transformer.py`
- Modify: `/Users/dgrauet/Work/ltx-2-mlx/tests/test_model_shapes.py`

**Why:** The TeaCache gate signal is block 0's modulated self-attention input (`video_normed` at line 266 of `transformer.py`). On skip steps the block doesn't run, so we factor the modulation into a small standalone helper that the model can call for any block (in practice, only block 0).

- [ ] **Step 1: Write the failing test**

Append this test to `/Users/dgrauet/Work/ltx-2-mlx/tests/test_model_shapes.py` (anywhere alongside other `BasicAVTransformerBlock` tests; if no such class exists, add a new `class TestBlockGateSignal`):

```python
class TestBlockGateSignal:
    def test_compute_video_normed_sa_matches_inline_modulation(self):
        """The helper must match the inline computation used at the start of
        the block's self-attention (transformer.py line 266)."""
        block = BasicAVTransformerBlock(
            video_dim=32, audio_dim=16,
            video_num_heads=4, audio_num_heads=4,
            video_head_dim=8, audio_head_dim=4,
            av_cross_num_heads=4, av_cross_head_dim=4,
            ff_mult=2.0,
        )
        B, Nv = 2, 6
        video_hidden = mx.random.normal((B, Nv, 32))
        # 9-param video AdaLN: (B, 9 * video_dim)
        video_adaln_params = mx.random.normal((B, 9 * 32))

        normed = block.compute_video_normed_sa(video_hidden, video_adaln_params)

        # Reference: replicate the transformer.py:266 computation
        v_shift_sa, v_scale_sa, *_ = block._unpack_adaln(
            video_adaln_params, block.scale_shift_table, 9, 32
        )
        ref = block._rms_norm(video_hidden) * (1.0 + v_scale_sa) + v_shift_sa

        mx.synchronize()
        assert normed.shape == (B, Nv, 32)
        assert mx.allclose(normed, ref, atol=1e-6).item()

    def test_compute_video_normed_sa_per_token_adaln(self):
        """Per-token AdaLN (B, N, 9*dim) must also work."""
        block = BasicAVTransformerBlock(
            video_dim=32, audio_dim=16,
            video_num_heads=4, audio_num_heads=4,
            video_head_dim=8, audio_head_dim=4,
            av_cross_num_heads=4, av_cross_head_dim=4,
            ff_mult=2.0,
        )
        B, Nv = 2, 6
        video_hidden = mx.random.normal((B, Nv, 32))
        video_adaln_params = mx.random.normal((B, Nv, 9 * 32))

        normed = block.compute_video_normed_sa(video_hidden, video_adaln_params)
        mx.synchronize()
        assert normed.shape == (B, Nv, 32)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd /Users/dgrauet/Work/ltx-2-mlx && python3 -m pytest tests/test_model_shapes.py::TestBlockGateSignal -v`

Expected: FAIL with `AttributeError: 'BasicAVTransformerBlock' object has no attribute 'compute_video_normed_sa'`.

- [ ] **Step 3: Implement the helper**

Open `/Users/dgrauet/Work/ltx-2-mlx/packages/ltx-core-mlx/src/ltx_core_mlx/model/transformer/transformer.py`. After the `_rms_norm` method (around line 180), insert:

```python
    def compute_video_normed_sa(
        self,
        video_hidden: mx.array,
        video_adaln_params: mx.array,
    ) -> mx.array:
        """Modulated video input to self-attention — the TeaCache gate signal.

        Mirrors the first two ops of ``__call__`` (the line that computes
        ``video_normed = rms(x) * (1 + scale_sa) + shift_sa``) without running
        attention. Used for cheap probe / cache-decision computations.

        Args:
            video_hidden: (B, Nv, video_dim) post-patchify hidden states.
            video_adaln_params: (B, 9*video_dim) or (B, Nv, 9*video_dim) AdaLN
                parameters for self-attn / ff / text-xattn (indices 0..2 are
                self-attn shift/scale/gate).

        Returns:
            Modulated input ``rms(video_hidden) * (1 + scale_sa) + shift_sa``,
            shape ``(B, Nv, video_dim)``.
        """
        vdim = video_hidden.shape[-1]
        v_shift_sa, v_scale_sa, *_ = self._unpack_adaln(
            video_adaln_params, self.scale_shift_table, 9, vdim
        )
        return self._rms_norm(video_hidden) * (1.0 + v_scale_sa) + v_shift_sa
```

- [ ] **Step 4: Run the tests**

Run: `cd /Users/dgrauet/Work/ltx-2-mlx && python3 -m pytest tests/test_model_shapes.py::TestBlockGateSignal -v`

Expected: 2 PASSED.

- [ ] **Step 5: Run ruff**

Run: `cd /Users/dgrauet/Work/ltx-2-mlx && python3 -m ruff check packages/ltx-core-mlx/src/ltx_core_mlx/model/transformer/transformer.py tests/test_model_shapes.py`

Expected: `All checks passed!`

- [ ] **Step 6: Commit**

```bash
cd /Users/dgrauet/Work/ltx-2-mlx
git add packages/ltx-core-mlx/src/ltx_core_mlx/model/transformer/transformer.py tests/test_model_shapes.py
git commit -m "$(cat <<'EOF'
feat: add compute_video_normed_sa helper for TeaCache gate signal

Factors the modulated-input computation (line 266 of __call__) into a
public helper. Used by LTXModel to compute block 0's gate signal cheaply
without running the block.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: `LTXModel.compute_gate_signal` — prelude-only entry point

**Files:**
- Modify: `/Users/dgrauet/Work/ltx-2-mlx/packages/ltx-core-mlx/src/ltx_core_mlx/model/transformer/model.py`
- Modify: `/Users/dgrauet/Work/ltx-2-mlx/tests/test_model_shapes.py`

**Why:** TeaCache decides skip-or-compute *before* invoking the model. The decision needs block 0's modulated input. We expose a method that runs only the prelude (patchify + AdaLN params + Task 2's helper) — much cheaper than a full forward.

- [ ] **Step 1: Write the failing test**

Append to `/Users/dgrauet/Work/ltx-2-mlx/tests/test_model_shapes.py` (alongside `TestBlockGateSignal`):

```python
class TestLTXModelGateSignal:
    @staticmethod
    def _tiny_config():
        return LTXModelConfig(
            num_layers=2,
            video_dim=32,
            audio_dim=16,
            video_num_heads=4,
            audio_num_heads=4,
            video_head_dim=8,
            audio_head_dim=4,
            av_cross_num_heads=4,
            av_cross_head_dim=4,
        )

    def test_compute_gate_signal_shape(self):
        model = LTXModel(self._tiny_config())
        B, Nv, Na = 2, 8, 4
        video_latent = mx.random.normal((B, Nv, 128))
        audio_latent = mx.random.normal((B, Na, 128))
        timestep = mx.array([0.5, 0.5], dtype=mx.bfloat16)

        gate = model.compute_gate_signal(video_latent, audio_latent, timestep)
        mx.synchronize()
        assert gate.shape == (B, Nv, 32)

    def test_gate_signal_matches_inline_block0_modulation(self):
        """The gate must equal block 0's video_normed during a real forward."""
        model = LTXModel(self._tiny_config())
        B, Nv, Na = 1, 4, 2
        video_latent = mx.random.normal((B, Nv, 128))
        audio_latent = mx.random.normal((B, Na, 128))
        timestep = mx.array([0.7], dtype=mx.bfloat16)

        gate = model.compute_gate_signal(video_latent, audio_latent, timestep)

        # Replicate the prelude manually and call block 0's helper.
        v_hidden = model.patchify_proj(video_latent.astype(mx.bfloat16))
        t_emb = model._embed_timestep_scalar(timestep.astype(mx.bfloat16))
        video_adaln_emb, _ = model.adaln_single(t_emb)
        ref = model.transformer_blocks[0].compute_video_normed_sa(v_hidden, video_adaln_emb)

        mx.synchronize()
        assert mx.allclose(gate, ref, atol=1e-5).item()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd /Users/dgrauet/Work/ltx-2-mlx && python3 -m pytest tests/test_model_shapes.py::TestLTXModelGateSignal -v`

Expected: FAIL with `AttributeError: 'LTXModel' object has no attribute 'compute_gate_signal'`.

- [ ] **Step 3: Implement `compute_gate_signal`**

Open `/Users/dgrauet/Work/ltx-2-mlx/packages/ltx-core-mlx/src/ltx_core_mlx/model/transformer/model.py`. Insert this method on `LTXModel` immediately after `_adaln_per_token` (around line 172, just before `__call__`):

```python
    def compute_gate_signal(
        self,
        video_latent: mx.array,
        audio_latent: mx.array,
        timestep: mx.array,
        video_timesteps: mx.array | None = None,
    ) -> mx.array:
        """Cheap probe: block 0's modulated video input (TeaCache gate signal).

        Runs the prelude (patchify_proj + video AdaLN) but no transformer
        blocks. The output is bit-equivalent to ``video_normed`` as it would
        be computed inside block 0 during a full forward.

        Args:
            video_latent: (B, Nv, video_patch_channels).
            audio_latent: (B, Na, audio_patch_channels). Unused for the
                gate signal itself but accepted for API symmetry; ignored.
            timestep: (B,) sigma value.
            video_timesteps: Optional (B, Nv) per-token timesteps; matches
                ``__call__`` semantics for conditioning masks.

        Returns:
            Gate signal ``(B, Nv, video_dim)``.
        """
        del audio_latent  # signature parity with __call__; not needed for gate
        video_latent = video_latent.astype(mx.bfloat16)
        timestep = timestep.astype(mx.bfloat16)

        video_hidden = self.patchify_proj(video_latent)
        t_emb = self._embed_timestep_scalar(timestep)

        if video_timesteps is not None:
            vt_emb = self._embed_timestep_per_token(video_timesteps)
            video_adaln_emb, _ = self._adaln_per_token(self.adaln_single, vt_emb)
        else:
            video_adaln_emb, _ = self.adaln_single(t_emb)

        return self.transformer_blocks[0].compute_video_normed_sa(video_hidden, video_adaln_emb)
```

- [ ] **Step 4: Run the tests**

Run: `cd /Users/dgrauet/Work/ltx-2-mlx && python3 -m pytest tests/test_model_shapes.py::TestLTXModelGateSignal -v`

Expected: 2 PASSED.

- [ ] **Step 5: Run ruff**

Run: `cd /Users/dgrauet/Work/ltx-2-mlx && python3 -m ruff check packages/ltx-core-mlx/src/ltx_core_mlx/model/transformer/model.py tests/test_model_shapes.py`

Expected: `All checks passed!`

- [ ] **Step 6: Commit**

```bash
cd /Users/dgrauet/Work/ltx-2-mlx
git add packages/ltx-core-mlx/src/ltx_core_mlx/model/transformer/model.py tests/test_model_shapes.py
git commit -m "$(cat <<'EOF'
feat: add LTXModel.compute_gate_signal for TeaCache decisions

Prelude-only forward that returns block 0's modulated input. Used by
the sampler to ask TeaCacheController.should_compute() before deciding
whether to run the full transformer.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: `LTXModel.__call__` — `tap` and `block_stack_override` kwargs

**Files:**
- Modify: `/Users/dgrauet/Work/ltx-2-mlx/packages/ltx-core-mlx/src/ltx_core_mlx/model/transformer/model.py`
- Modify: `/Users/dgrauet/Work/ltx-2-mlx/tests/test_model_shapes.py`

**Why:** Two opt-in hooks. `tap` is observation-only (calibration). `block_stack_override` lets callers (the sampler, on a TeaCache skip) replace the block iteration with a cached-residual reconstruction. Both default `None` for zero overhead.

- [ ] **Step 1: Write the failing tests**

Append to `/Users/dgrauet/Work/ltx-2-mlx/tests/test_model_shapes.py` (after `TestLTXModelGateSignal`):

```python
class TestLTXModelHooks:
    @staticmethod
    def _tiny():
        return LTXModel(
            LTXModelConfig(
                num_layers=2, video_dim=32, audio_dim=16,
                video_num_heads=4, audio_num_heads=4,
                video_head_dim=8, audio_head_dim=4,
                av_cross_num_heads=4, av_cross_head_dim=4,
            )
        )

    def _inputs(self):
        B, Nv, Na = 1, 4, 2
        return {
            "video_latent": mx.random.normal((B, Nv, 128)),
            "audio_latent": mx.random.normal((B, Na, 128)),
            "timestep": mx.array([0.5], dtype=mx.bfloat16),
        }

    def test_tap_called_once_with_block_residuals(self):
        model = self._tiny()
        recorded = []

        def tap(video_block_residual, audio_block_residual):
            recorded.append((video_block_residual.shape, audio_block_residual.shape))

        out_v, out_a = model(**self._inputs(), tap=tap)
        mx.synchronize()
        assert len(recorded) == 1
        assert recorded[0] == ((1, 4, 32), (1, 2, 16))
        assert out_v.shape == (1, 4, 128)
        assert out_a.shape == (1, 2, 128)

    def test_no_tap_no_overhead_no_failure(self):
        model = self._tiny()
        out_v, out_a = model(**self._inputs())
        mx.synchronize()
        assert out_v.shape == (1, 4, 128)
        assert out_a.shape == (1, 2, 128)

    def test_block_stack_override_replaces_block_iteration(self):
        model = self._tiny()
        called = []

        def override(v_hidden, a_hidden):
            called.append(True)
            # Return a fixed deterministic transformation
            return v_hidden + 1.0, a_hidden - 1.0

        out_v, out_a = model(**self._inputs(), block_stack_override=override)
        mx.synchronize()
        assert called == [True]
        assert out_v.shape == (1, 4, 128)
        # Sanity: the head still ran (output channels are patch_channels=128, not video_dim=32)

    def test_override_and_tap_coexist(self):
        model = self._tiny()
        recorded = []

        def tap(v_res, a_res):
            recorded.append((v_res.shape, a_res.shape))

        def override(v_hidden, a_hidden):
            return v_hidden * 2.0, a_hidden * 2.0

        model(**self._inputs(), tap=tap, block_stack_override=override)
        mx.synchronize()
        # Tap fires once; residual is computed against the override output.
        assert len(recorded) == 1
        # Residuals are (override_output - block_input) — non-zero by construction.
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd /Users/dgrauet/Work/ltx-2-mlx && python3 -m pytest tests/test_model_shapes.py::TestLTXModelHooks -v`

Expected: FAIL with `TypeError: __call__() got an unexpected keyword argument 'tap'`.

- [ ] **Step 3: Plumb the kwargs through `LTXModel.__call__`**

Open `/Users/dgrauet/Work/ltx-2-mlx/packages/ltx-core-mlx/src/ltx_core_mlx/model/transformer/model.py`.

In the `__call__` signature (currently ending at line 187 with `perturbations: BatchedPerturbationConfig | None = None,`), add after `perturbations`:

```python
        tap: callable | None = None,
        block_stack_override: callable | None = None,
```

Update the docstring's `Args:` block to document them (insert after the `perturbations` doc):

```
            tap: Optional callback ``tap(video_block_residual,
                audio_block_residual)`` invoked after the block stack with
                the residuals ``block_output - block_input``. Used for
                calibration; does not affect control flow.
            block_stack_override: Optional callable
                ``(video_hidden, audio_hidden) -> (video_hidden_out,
                audio_hidden_out)`` that replaces the block iteration. Used
                by TeaCache on skip steps to reconstruct outputs from a
                cached residual without running the blocks. The model's
                head still runs.
```

Now modify the block iteration. Find the loop starting at line 307 and the head starting at line 332. Replace the block iteration + head section with:

```python
        # --- Block stack (optionally overridden) ---
        block_input_v = video_hidden
        block_input_a = audio_hidden

        if block_stack_override is not None:
            video_hidden, audio_hidden = block_stack_override(video_hidden, audio_hidden)
        else:
            for block_idx, block in enumerate(self.transformer_blocks):
                video_hidden, audio_hidden = block(
                    video_hidden=video_hidden,
                    audio_hidden=audio_hidden,
                    video_adaln_params=video_adaln_emb,
                    audio_adaln_params=audio_adaln_emb,
                    video_prompt_adaln_params=video_prompt_emb,
                    audio_prompt_adaln_params=audio_prompt_emb,
                    av_ca_video_params=av_ca_video_emb,
                    av_ca_audio_params=av_ca_audio_emb,
                    av_ca_a2v_gate_params=av_ca_a2v_gate_emb,
                    av_ca_v2a_gate_params=av_ca_v2a_gate_emb,
                    video_text_embeds=video_text_embeds,
                    audio_text_embeds=audio_text_embeds,
                    video_rope_freqs=video_rope_freqs,
                    audio_rope_freqs=audio_rope_freqs,
                    video_cross_rope_freqs=video_cross_rope_freqs,
                    audio_cross_rope_freqs=audio_cross_rope_freqs,
                    video_attention_mask=video_attention_mask,
                    audio_attention_mask=audio_attention_mask,
                    perturbations=perturbations,
                    block_idx=block_idx,
                )

        if tap is not None:
            tap(video_hidden - block_input_v, audio_hidden - block_input_a)

        # Output: AdaLN with scale_shift_table + embedded_timestep + proj
        video_out = self._output_block(video_hidden, video_embedded_ts, self.scale_shift_table, self.proj_out)
        audio_out = self._output_block(
            audio_hidden, audio_embedded_ts, self.audio_scale_shift_table, self.audio_proj_out
        )

        return video_out, audio_out
```

- [ ] **Step 4: Plumb the kwargs through `X0Model.__call__`**

Still in `model.py`, find `X0Model.__call__` (around line 411). Add the two kwargs to the signature, after `perturbations`:

```python
        tap: callable | None = None,
        block_stack_override: callable | None = None,
```

Pass them through in the inner call to `self.model(...)` (around line 438):

```python
        video_v, audio_v = self.model(
            video_latent=video_latent,
            audio_latent=audio_latent,
            timestep=sigma,
            video_timesteps=video_timesteps,
            audio_timesteps=audio_timesteps,
            perturbations=perturbations,
            tap=tap,
            block_stack_override=block_stack_override,
            **kwargs,
        )
```

- [ ] **Step 5: Run the tests**

Run: `cd /Users/dgrauet/Work/ltx-2-mlx && python3 -m pytest tests/test_model_shapes.py::TestLTXModelHooks -v`

Expected: 4 PASSED.

- [ ] **Step 6: Verify no existing tests broke**

Run: `cd /Users/dgrauet/Work/ltx-2-mlx && python3 -m pytest tests/test_model_shapes.py -q`

Expected: all PASSED (existing + 4 new).

- [ ] **Step 7: Run ruff**

Run: `cd /Users/dgrauet/Work/ltx-2-mlx && python3 -m ruff check packages/ltx-core-mlx/src/ltx_core_mlx/model/transformer/model.py tests/test_model_shapes.py`

Expected: `All checks passed!`

- [ ] **Step 8: Commit**

```bash
cd /Users/dgrauet/Work/ltx-2-mlx
git add packages/ltx-core-mlx/src/ltx_core_mlx/model/transformer/model.py tests/test_model_shapes.py
git commit -m "$(cat <<'EOF'
feat: add tap and block_stack_override hooks to LTXModel and X0Model

Two opt-in hooks default to None (zero overhead): tap is a pure
observation callback receiving (video_block_residual, audio_block_residual)
after the block stack; block_stack_override replaces the block iteration
and is used by TeaCache on skip steps to reconstruct outputs from a
cached residual without invoking the blocks.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: `guided_denoise_loop` — `tap` kwarg

**Files:**
- Modify: `/Users/dgrauet/Work/ltx-2-mlx/packages/ltx-pipelines-mlx/src/ltx_pipelines_mlx/utils/samplers.py`
- Create: `/Users/dgrauet/Work/ltx-2-mlx/tests/test_teacache_integration.py`

**Why:** The calibration script needs `(step_idx, gate_signal, video_residual, audio_residual)` per step. `tap` plumbs that out of the conditioned pass.

- [ ] **Step 1: Write the failing test**

Create `/Users/dgrauet/Work/ltx-2-mlx/tests/test_teacache_integration.py`:

```python
"""Integration tests for the sampler-level TeaCache hooks (tap + teacache)."""

from __future__ import annotations

import mlx.core as mx
import pytest

from ltx_core_mlx.components.guiders import (
    MultiModalGuiderFactory,
    MultiModalGuiderParams,
    create_multimodal_guider_factory,
)
from ltx_core_mlx.conditioning.types.latent_cond import LatentState
from ltx_core_mlx.model.transformer.model import LTXModel, LTXModelConfig, X0Model
from ltx_pipelines_mlx.utils.samplers import guided_denoise_loop


def _tiny_x0_model():
    return X0Model(
        LTXModel(
            LTXModelConfig(
                num_layers=2, video_dim=32, audio_dim=16,
                video_num_heads=4, audio_num_heads=4,
                video_head_dim=8, audio_head_dim=4,
                av_cross_num_heads=4, av_cross_head_dim=4,
            )
        )
    )


def _make_state(B: int, N: int) -> LatentState:
    latent = mx.random.normal((B, N, 128)).astype(mx.bfloat16)
    return LatentState(
        latent=latent,
        clean_latent=mx.zeros_like(latent),
        denoise_mask=mx.ones((B, N), dtype=mx.bfloat16),
    )


def _cfg_factory(B: int, video_dim: int) -> MultiModalGuiderFactory:
    """CFG-only guider factory (no STG, no modality)."""
    params = MultiModalGuiderParams(
        cfg_scale=3.0, stg_scale=0.0, rescale_scale=0.7,
        modality_scale=1.0, stg_blocks=[],
    )
    neg = mx.zeros((B, 4, video_dim), dtype=mx.bfloat16)
    return create_multimodal_guider_factory(params, negative_context=neg)


class TestTapHook:
    def test_tap_fires_once_per_step(self):
        model = _tiny_x0_model()
        B, Nv, Na = 1, 4, 2
        video_state = _make_state(B, Nv)
        audio_state = _make_state(B, Na)

        # 4 steps total → 3 sigma pairs (denoising)
        sigmas = [1.0, 0.7, 0.4, 0.1, 0.0]
        video_text_embeds = mx.zeros((B, 4, 32), dtype=mx.bfloat16)
        audio_text_embeds = mx.zeros((B, 4, 16), dtype=mx.bfloat16)

        recorded: list[tuple] = []

        def tap(step_idx, gate, video_residual, audio_residual):
            recorded.append((step_idx, gate.shape, video_residual.shape, audio_residual.shape))

        guided_denoise_loop(
            model=model,
            video_state=video_state,
            audio_state=audio_state,
            video_text_embeds=video_text_embeds,
            audio_text_embeds=audio_text_embeds,
            video_guider_factory=_cfg_factory(B, 32),
            audio_guider_factory=_cfg_factory(B, 16),
            sigmas=sigmas,
            show_progress=False,
            tap=tap,
        )
        assert [r[0] for r in recorded] == [0, 1, 2, 3]
        for _, gate_shape, v_res_shape, a_res_shape in recorded:
            assert gate_shape == (B, Nv, 32)
            assert v_res_shape == (B, Nv, 32)
            assert a_res_shape == (B, Na, 16)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd /Users/dgrauet/Work/ltx-2-mlx && python3 -m pytest tests/test_teacache_integration.py::TestTapHook -v`

Expected: FAIL with `TypeError: guided_denoise_loop() got an unexpected keyword argument 'tap'`.

- [ ] **Step 3: Add `tap` to `guided_denoise_loop`**

Open `/Users/dgrauet/Work/ltx-2-mlx/packages/ltx-pipelines-mlx/src/ltx_pipelines_mlx/utils/samplers.py`.

Add to the `guided_denoise_loop` signature (currently ends at line 521 with `show_progress: bool = True,`):

```python
    tap: callable | None = None,
```

Update the docstring `Args:` block (insert before the closing `Returns:`):

```
        tap: Optional per-step instrumentation hook called as
            ``tap(step_idx, gate_signal, video_block_residual,
            audio_block_residual)`` after the conditioned-pass forward.
            Used by the TeaCache calibration script. Has no effect on
            control flow.
```

Inside the per-step loop (around line 635, just before the line `cond_video_x0, cond_audio_x0 = model(**cond_kwargs)`), add the tap-aware wiring. Replace the line:

```python
        cond_video_x0, cond_audio_x0 = model(**cond_kwargs)
```

with:

```python
        if tap is not None:
            gate_signal = model.model.compute_gate_signal(
                video_latent=video_x,
                audio_latent=audio_x,
                timestep=mx.broadcast_to(sigma_arr, (B,)),
                video_timesteps=base_kwargs.get("video_timesteps"),
            )
            captured: dict = {}

            def _capture_tap(v_res, a_res):
                captured["v"] = v_res
                captured["a"] = a_res

            cond_video_x0, cond_audio_x0 = model(**cond_kwargs, tap=_capture_tap)
            tap(step_idx, gate_signal, captured["v"], captured["a"])
        else:
            cond_video_x0, cond_audio_x0 = model(**cond_kwargs)
```

(`model.model` is the inner `LTXModel`, since `model` here is the wrapping `X0Model`.)

- [ ] **Step 4: Run the test**

Run: `cd /Users/dgrauet/Work/ltx-2-mlx && python3 -m pytest tests/test_teacache_integration.py::TestTapHook -v`

Expected: PASS.

- [ ] **Step 5: Verify no existing sampler tests broke**

Run: `cd /Users/dgrauet/Work/ltx-2-mlx && python3 -m pytest tests/test_denoise.py tests/test_guidance.py -q`

Expected: all PASS.

- [ ] **Step 6: Run ruff**

Run: `cd /Users/dgrauet/Work/ltx-2-mlx && python3 -m ruff check packages/ltx-pipelines-mlx/src/ltx_pipelines_mlx/utils/samplers.py tests/test_teacache_integration.py`

Expected: `All checks passed!`

- [ ] **Step 7: Commit**

```bash
cd /Users/dgrauet/Work/ltx-2-mlx
git add packages/ltx-pipelines-mlx/src/ltx_pipelines_mlx/utils/samplers.py tests/test_teacache_integration.py
git commit -m "$(cat <<'EOF'
feat: add tap kwarg to guided_denoise_loop for TeaCache calibration

The conditioned pass exposes (step_idx, gate_signal, video_residual,
audio_residual) per step via the tap callback. Used by the calibration
script; no effect when tap=None.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: `guided_denoise_loop` — `teacache` kwarg with multi-pass dict cache

**Files:**
- Modify: `/Users/dgrauet/Work/ltx-2-mlx/packages/ltx-pipelines-mlx/src/ltx_pipelines_mlx/utils/samplers.py`
- Modify: `/Users/dgrauet/Work/ltx-2-mlx/tests/test_teacache_integration.py`

**Why:** The production-side hook. Owns the per-step decision and the per-pass residual dict.

- [ ] **Step 1: Write the failing tests**

Append to `/Users/dgrauet/Work/ltx-2-mlx/tests/test_teacache_integration.py`:

```python
class _StubController:
    """Records calls and returns canned should_compute decisions."""

    def __init__(self, decisions: list[bool]):
        self.decisions = list(decisions)
        self.cached = None
        self.gate_calls: list = []
        self.cache_calls: list = []
        self.reset_calls = 0

    def should_compute(self, step_idx, gate_signal):
        self.gate_calls.append((step_idx, gate_signal.shape))
        return self.decisions.pop(0)

    def cache_residual(self, payload):
        self.cache_calls.append(payload)
        self.cached = payload

    @property
    def previous_residual(self):
        if self.cached is None:
            raise RuntimeError("nothing cached")
        return self.cached

    def reset(self):
        self.reset_calls += 1


class TestTeaCacheHook:
    def _run(self, decisions: list[bool]):
        model = _tiny_x0_model()
        B, Nv, Na = 1, 4, 2
        video_state = _make_state(B, Nv)
        audio_state = _make_state(B, Na)
        sigmas = [1.0, 0.7, 0.4, 0.1, 0.0]
        controller = _StubController(decisions)
        guided_denoise_loop(
            model=model,
            video_state=video_state,
            audio_state=audio_state,
            video_text_embeds=mx.zeros((B, 4, 32), dtype=mx.bfloat16),
            audio_text_embeds=mx.zeros((B, 4, 16), dtype=mx.bfloat16),
            video_guider_factory=_cfg_factory(B, 32),
            audio_guider_factory=_cfg_factory(B, 16),
            sigmas=sigmas,
            show_progress=False,
            teacache=controller,
        )
        return controller

    def test_should_compute_called_once_per_step(self):
        c = self._run(decisions=[True, True, True, True])
        assert len(c.gate_calls) == 4
        assert all(shape == (1, 4, 32) for _, shape in c.gate_calls)

    def test_compute_path_caches_per_pass_dict(self):
        c = self._run(decisions=[True, True, True, True])
        # 4 cache_residual calls (one per step). Each is a dict with 'cond' and 'uncond' keys.
        assert len(c.cache_calls) == 4
        for payload in c.cache_calls:
            assert isinstance(payload, dict)
            assert set(payload.keys()) == {"cond", "uncond"}
            v_res, a_res = payload["cond"]
            assert v_res.shape == (1, 4, 32)
            assert a_res.shape == (1, 2, 16)

    def test_skip_path_does_not_cache(self):
        """When should_compute returns False at step 1, no cache is stored
        for that step (the previous residual is reused via override). 4 steps
        with one skip → 3 cache_residual calls."""
        c = self._run(decisions=[True, False, True, True])
        assert len(c.gate_calls) == 4
        assert len(c.cache_calls) == 3
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd /Users/dgrauet/Work/ltx-2-mlx && python3 -m pytest tests/test_teacache_integration.py::TestTeaCacheHook -v`

Expected: FAIL with `TypeError: guided_denoise_loop() got an unexpected keyword argument 'teacache'`.

- [ ] **Step 3: Add `teacache` to `guided_denoise_loop`**

Open `/Users/dgrauet/Work/ltx-2-mlx/packages/ltx-pipelines-mlx/src/ltx_pipelines_mlx/utils/samplers.py`.

Add to the signature, after `tap`:

```python
    teacache=None,  # mlx_arsenal.diffusion.TeaCacheController-compatible
```

Update the docstring (insert before `Returns:`):

```
        teacache: Optional TeaCache controller. When provided, the loop
            calls ``teacache.should_compute(step_idx, gate_signal)`` once
            per step using block 0's modulated input from the conditioned
            pass; on True, all guidance passes run normally and their
            block residuals are cached as a dict keyed by pass label
            (``cond``, ``uncond``, ``ptb``, ``mod``); on False, the
            previous step's cached residuals replace the block stack via
            ``block_stack_override``. The transformer head still runs
            on every pass.
```

Now the heart of the change. Replace the four guidance passes (lines ~629-706) with a refactored version that funnels each pass through a helper. Find:

```python
        # --- 1. Conditioned prediction (positive context) ---
        cond_kwargs = {
            **base_kwargs,
            "video_text_embeds": video_text_embeds,
            "audio_text_embeds": audio_text_embeds,
        }
        cond_video_x0, cond_audio_x0 = model(**cond_kwargs)
```

…and the surrounding three other pass blocks (uncond at ~637-655, ptb at ~657-686, mod at ~688-706). Replace the entire span from `# --- 1. Conditioned prediction` through the end of `# --- 4. Isolated modality prediction` (just before `# --- 5. Apply guiders ---`) with:

```python
        # --- Compute the gate signal once per step (cheap, runs prelude only).
        # Used both by tap and by teacache decisions. None when neither hook needs it.
        gate_signal = None
        if tap is not None or teacache is not None:
            gate_signal = model.model.compute_gate_signal(
                video_latent=video_x,
                audio_latent=audio_x,
                timestep=mx.broadcast_to(sigma_arr, (B,)),
                video_timesteps=base_kwargs.get("video_timesteps"),
            )

        should_compute_full = True
        if teacache is not None:
            should_compute_full = teacache.should_compute(step_idx, gate_signal)

        # Pass-level helpers: capture residual on compute path, override on skip.
        captured_residuals: dict = {}
        cached_residuals = teacache.previous_residual if (teacache is not None and not should_compute_full) else None

        def _make_capture_tap(label: str):
            def _t(v_res, a_res):
                captured_residuals[label] = (v_res, a_res)
            return _t

        def _make_override(label: str):
            v_res, a_res = cached_residuals[label]
            def _o(v_hidden, a_hidden):
                return v_hidden + v_res, a_hidden + a_res
            return _o

        def _run_pass(label: str, kwargs: dict):
            if should_compute_full:
                # cond pass also drives the calibration tap
                pass_tap = _make_capture_tap(label) if (tap is not None or teacache is not None) else None
                return model(**kwargs, tap=pass_tap)
            else:
                return model(**kwargs, block_stack_override=_make_override(label))

        # --- 1. Conditioned prediction (positive context) ---
        cond_kwargs = {
            **base_kwargs,
            "video_text_embeds": video_text_embeds,
            "audio_text_embeds": audio_text_embeds,
        }
        cond_video_x0, cond_audio_x0 = _run_pass("cond", cond_kwargs)

        if tap is not None and "cond" in captured_residuals:
            v_res, a_res = captured_residuals["cond"]
            tap(step_idx, gate_signal, v_res, a_res)

        # --- 2. Unconditional prediction for CFG ---
        neg_video_x0: mx.array | float = 0.0
        neg_audio_x0: mx.array | float = 0.0

        if video_guider.do_unconditional_generation() or audio_guider.do_unconditional_generation():
            neg_video_embeds = (
                video_guider.negative_context if video_guider.negative_context is not None else video_text_embeds
            )
            neg_audio_embeds = (
                audio_guider.negative_context if audio_guider.negative_context is not None else audio_text_embeds
            )

            neg_kwargs = {
                **base_kwargs,
                "video_text_embeds": neg_video_embeds,
                "audio_text_embeds": neg_audio_embeds,
            }
            neg_video_x0, neg_audio_x0 = _run_pass("uncond", neg_kwargs)

        # --- 3. Perturbed prediction for STG ---
        ptb_video_x0: mx.array | float = 0.0
        ptb_audio_x0: mx.array | float = 0.0

        if video_guider.do_perturbed_generation() or audio_guider.do_perturbed_generation():
            perturbation_list: list[Perturbation] = []
            if video_guider.do_perturbed_generation():
                perturbation_list.append(
                    Perturbation(
                        type=PerturbationType.SKIP_VIDEO_SELF_ATTN,
                        blocks=video_guider.params.stg_blocks,
                    )
                )
            if audio_guider.do_perturbed_generation():
                perturbation_list.append(
                    Perturbation(
                        type=PerturbationType.SKIP_AUDIO_SELF_ATTN,
                        blocks=audio_guider.params.stg_blocks,
                    )
                )
            perturbation_config = PerturbationConfig(perturbations=perturbation_list)
            batched_perturbations = BatchedPerturbationConfig(perturbations=[perturbation_config] * B)

            ptb_kwargs = {
                **base_kwargs,
                "video_text_embeds": video_text_embeds,
                "audio_text_embeds": audio_text_embeds,
                "perturbations": batched_perturbations,
            }
            ptb_video_x0, ptb_audio_x0 = _run_pass("ptb", ptb_kwargs)

        # --- 4. Isolated modality prediction ---
        mod_video_x0: mx.array | float = 0.0
        mod_audio_x0: mx.array | float = 0.0

        if video_guider.do_isolated_modality_generation() or audio_guider.do_isolated_modality_generation():
            mod_perturbation_list = [
                Perturbation(type=PerturbationType.SKIP_A2V_CROSS_ATTN, blocks=None),
                Perturbation(type=PerturbationType.SKIP_V2A_CROSS_ATTN, blocks=None),
            ]
            mod_perturbation_config = PerturbationConfig(perturbations=mod_perturbation_list)
            mod_batched_perturbations = BatchedPerturbationConfig(perturbations=[mod_perturbation_config] * B)

            mod_kwargs = {
                **base_kwargs,
                "video_text_embeds": video_text_embeds,
                "audio_text_embeds": audio_text_embeds,
                "perturbations": mod_batched_perturbations,
            }
            mod_video_x0, mod_audio_x0 = _run_pass("mod", mod_kwargs)

        if teacache is not None and should_compute_full:
            teacache.cache_residual(captured_residuals)
```

Note: this also removes the duplicate `tap` logic from Task 5 — the helper `_run_pass` now drives capture for both `tap` and `teacache`. After the replacement, **delete the Task 5 wiring** (the old `if tap is not None: gate_signal = ...` block above the cond pass that you added in Task 5). The test `TestTapHook::test_tap_fires_once_per_step` from Task 5 still passes via the new path.

- [ ] **Step 4: Run all teacache integration tests**

Run: `cd /Users/dgrauet/Work/ltx-2-mlx && python3 -m pytest tests/test_teacache_integration.py -v`

Expected: 4 PASSED (`TestTapHook::test_tap_fires_once_per_step` + 3 in `TestTeaCacheHook`).

- [ ] **Step 5: Run all existing sampler tests**

Run: `cd /Users/dgrauet/Work/ltx-2-mlx && python3 -m pytest tests/test_denoise.py tests/test_guidance.py tests/test_two_stage.py -q`

Expected: all PASS.

- [ ] **Step 6: Run ruff**

Run: `cd /Users/dgrauet/Work/ltx-2-mlx && python3 -m ruff check packages/ltx-pipelines-mlx/src/ltx_pipelines_mlx/utils/samplers.py tests/test_teacache_integration.py`

Expected: `All checks passed!`

- [ ] **Step 7: Commit**

```bash
cd /Users/dgrauet/Work/ltx-2-mlx
git add packages/ltx-pipelines-mlx/src/ltx_pipelines_mlx/utils/samplers.py tests/test_teacache_integration.py
git commit -m "$(cat <<'EOF'
feat: add teacache kwarg to guided_denoise_loop with per-pass dict cache

The conditioned pass's gate signal drives the once-per-step
should_compute decision. On compute, all guidance passes run normally
and their block residuals are cached as a dict keyed by pass label. On
skip, block_stack_override reuses each pass's cached residual; the
block stack is not invoked. tap continues to fire once per step on the
conditioned pass, sharing the gate signal computation.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: `TwoStagePipeline.generate_two_stage` — `enable_teacache` flag

**Files:**
- Modify: `/Users/dgrauet/Work/ltx-2-mlx/packages/ltx-pipelines-mlx/src/ltx_pipelines_mlx/ti2vid_two_stages.py`
- Modify: `/Users/dgrauet/Work/ltx-2-mlx/tests/test_teacache_integration.py`

**Why:** Public-facing opt-in flag. Default off; on enable, instantiate the controller from the `ltx2` arsenal preset and pass it down.

- [ ] **Step 1: Write the failing test**

Append to `/Users/dgrauet/Work/ltx-2-mlx/tests/test_teacache_integration.py`:

```python
class TestPipelineFlag:
    def test_enable_teacache_raises_clear_error_when_preset_missing(self):
        """Until calibration is run, the ltx2 preset is absent. Pipeline must
        raise a clear error pointing at the calibration script (not a generic
        KeyError)."""
        from mlx_arsenal.diffusion import TEACACHE_PRESETS

        # Fail fast if a future change accidentally adds the preset before this
        # test is removed.
        if "ltx2" in TEACACHE_PRESETS:
            pytest.skip("ltx2 preset already present; this guard test no longer applies")

        from ltx_pipelines_mlx.ti2vid_two_stages import _build_teacache_controller

        with pytest.raises(RuntimeError, match="calibrate_teacache"):
            _build_teacache_controller(num_steps=30, thresh=None)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd /Users/dgrauet/Work/ltx-2-mlx && python3 -m pytest tests/test_teacache_integration.py::TestPipelineFlag -v`

Expected: FAIL with `ImportError: cannot import name '_build_teacache_controller'`.

- [ ] **Step 3: Add the controller builder + the kwargs**

Open `/Users/dgrauet/Work/ltx-2-mlx/packages/ltx-pipelines-mlx/src/ltx_pipelines_mlx/ti2vid_two_stages.py`.

Just below the existing imports (after the line importing from `samplers`), add:

```python
from mlx_arsenal.diffusion import TEACACHE_PRESETS, TeaCacheController


def _build_teacache_controller(num_steps: int, thresh: float | None) -> TeaCacheController:
    """Construct a controller for LTX-2 stage 1; raise if uncalibrated.

    Args:
        num_steps: Number of denoising steps for stage 1.
        thresh: Optional override for the preset's default ``rel_l1_thresh``.

    Returns:
        Configured ``TeaCacheController``.

    Raises:
        RuntimeError: If the ``ltx2`` preset is missing — calibration not done.
    """
    if "ltx2" not in TEACACHE_PRESETS:
        raise RuntimeError(
            "TeaCache preset 'ltx2' is missing — run "
            "scripts/calibrate_teacache.py to generate coefficients, then "
            "add the snippet to mlx_arsenal/diffusion/teacache.py "
            "TEACACHE_PRESETS."
        )
    return TeaCacheController.from_preset("ltx2", num_steps=num_steps, rel_l1_thresh=thresh)
```

Now find `generate_two_stage` (signature begins at line 161). Add to the kwargs (after `audio_guider_params`):

```python
        enable_teacache: bool = False,
        teacache_thresh: float | None = None,
```

Update the docstring `Args:` block to describe them (insert before `Returns:`):

```
            enable_teacache: When True, instantiate a TeaCacheController
                from the 'ltx2' arsenal preset and use it to skip stage 1
                transformer forwards whose modulated input is sufficiently
                close to the previous step's. Default False (no caching).
            teacache_thresh: Optional override for the preset's default
                ``rel_l1_thresh``. Higher = more skipping = faster but
                lossier. Ignored when ``enable_teacache=False``.
```

In the body of `generate_two_stage`, just before the `output_1 = guided_denoise_loop(...)` call (around line 271), build the controller conditionally:

```python
        teacache_controller = None
        if enable_teacache:
            teacache_controller = _build_teacache_controller(stage1_steps, teacache_thresh)
            teacache_controller.reset()
```

And pass it into the call by adding a `teacache=teacache_controller` kwarg to the `guided_denoise_loop(...)` invocation:

```python
        output_1 = guided_denoise_loop(
            model=x0_model,
            video_state=video_state,
            audio_state=audio_state,
            video_text_embeds=video_embeds,
            audio_text_embeds=audio_embeds,
            video_guider_factory=video_factory,
            audio_guider_factory=audio_factory,
            sigmas=sigmas_1,
            teacache=teacache_controller,
        )
```

- [ ] **Step 4: Run the tests**

Run: `cd /Users/dgrauet/Work/ltx-2-mlx && python3 -m pytest tests/test_teacache_integration.py::TestPipelineFlag -v`

Expected: PASS.

- [ ] **Step 5: Verify two-stage tests still pass**

Run: `cd /Users/dgrauet/Work/ltx-2-mlx && python3 -m pytest tests/test_two_stage.py -q`

Expected: all PASS.

- [ ] **Step 6: Run ruff**

Run: `cd /Users/dgrauet/Work/ltx-2-mlx && python3 -m ruff check packages/ltx-pipelines-mlx/src/ltx_pipelines_mlx/ti2vid_two_stages.py tests/test_teacache_integration.py`

Expected: `All checks passed!`

- [ ] **Step 7: Commit**

```bash
cd /Users/dgrauet/Work/ltx-2-mlx
git add packages/ltx-pipelines-mlx/src/ltx_pipelines_mlx/ti2vid_two_stages.py tests/test_teacache_integration.py
git commit -m "$(cat <<'EOF'
feat: add enable_teacache flag to TwoStagePipeline.generate_two_stage

Default off. When True, builds a TeaCacheController from the arsenal
'ltx2' preset (raises RuntimeError if absent, pointing at the
calibration script) and threads it into guided_denoise_loop. Optional
teacache_thresh kwarg overrides the preset default.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 8: Calibration script `scripts/calibrate_teacache.py`

**Files:**
- Create: `/Users/dgrauet/Work/ltx-2-mlx/packages/ltx-pipelines-mlx/scripts/calibrate_teacache.py`
- Create: `/Users/dgrauet/Work/ltx-2-mlx/packages/ltx-pipelines-mlx/scripts/__init__.py` (if missing)

**Why:** The artifact the user runs on their 32 GB host to produce the LTX-2 coefficients.

This task does NOT include unit tests — the script orchestrates a real model run, which we can't execute here. Validation is delegated to the user. The polyfit math is small and can be unit-tested separately if desired (see "Open follow-ups" at the bottom of this plan).

- [ ] **Step 1: Ensure the scripts package exists**

Run:

```bash
ls /Users/dgrauet/Work/ltx-2-mlx/packages/ltx-pipelines-mlx/scripts/__init__.py 2>/dev/null || \
    touch /Users/dgrauet/Work/ltx-2-mlx/packages/ltx-pipelines-mlx/scripts/__init__.py
```

- [ ] **Step 2: Plumb `tap` through `generate_two_stage`**

Open `/Users/dgrauet/Work/ltx-2-mlx/packages/ltx-pipelines-mlx/src/ltx_pipelines_mlx/ti2vid_two_stages.py`.

Add `tap: callable | None = None` to the `generate_two_stage` signature (next to `enable_teacache` from Task 7). Update the docstring `Args:` block (insert before `Returns:`):

```
            tap: Optional per-step instrumentation hook forwarded to
                ``guided_denoise_loop``. Used by the calibration script.
```

Pass it through in the `guided_denoise_loop(...)` call (add to the kwargs alongside `teacache=`):

```python
        output_1 = guided_denoise_loop(
            model=x0_model,
            video_state=video_state,
            audio_state=audio_state,
            video_text_embeds=video_embeds,
            audio_text_embeds=audio_embeds,
            video_guider_factory=video_factory,
            audio_guider_factory=audio_factory,
            sigmas=sigmas_1,
            teacache=teacache_controller,
            tap=tap,
        )
```

- [ ] **Step 3: Write the calibration script**

Create `/Users/dgrauet/Work/ltx-2-mlx/packages/ltx-pipelines-mlx/scripts/calibrate_teacache.py`:

```python
"""Calibrate TeaCache polyfit coefficients for LTX-2 stage 1.

Runs N reference generations through ``TwoStagePipeline.generate_two_stage``
with caching disabled, captures per-step (gate_signal, block_residual) for
the conditioned pass, computes input/output L1 deltas in fp32, and fits a
degree-4 polynomial mapping input delta → output delta. Writes the
coefficients (and a copy-pasteable preset snippet) to disk.

Memory: rolling per-step state only (no per-step buffering of full tensors).
Suitable for 32 GB hosts running the q8 model + Gemma text encoder.

Usage:
    python -m ltx_pipelines_mlx.scripts.calibrate_teacache \\
        --model-dir dgrauet/ltx-2.3-mlx-q8 \\
        --prompts prompts.txt \\
        --num-steps 30 \\
        --out coefficients.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np

from ltx_pipelines_mlx.ti2vid_two_stages import TwoStagePipeline


def _rel_l1_fp32(curr: mx.array, prev: mx.array) -> float:
    """Relative L1 distance with fp32 reductions: mean|curr-prev| / mean|prev|."""
    curr32 = curr.astype(mx.float32)
    prev32 = prev.astype(mx.float32)
    diff = mx.mean(mx.abs(curr32 - prev32))
    base = mx.mean(mx.abs(prev32))
    return float((diff / base).item())


class _StreamingCalibrator:
    """Rolls (prev_modulated, prev_residual) per stream and accumulates deltas."""

    def __init__(self):
        self._prev_gate: mx.array | None = None
        self._prev_v_res: mx.array | None = None
        self._prev_a_res: mx.array | None = None
        self.delta_in: list[float] = []
        self.delta_res_video: list[float] = []
        self.delta_res_audio: list[float] = []

    def __call__(self, step_idx: int, gate: mx.array, v_res: mx.array, a_res: mx.array) -> None:
        if self._prev_gate is not None:
            self.delta_in.append(_rel_l1_fp32(gate, self._prev_gate))
            self.delta_res_video.append(_rel_l1_fp32(v_res, self._prev_v_res))
            self.delta_res_audio.append(_rel_l1_fp32(a_res, self._prev_a_res))
        self._prev_gate = gate
        self._prev_v_res = v_res
        self._prev_a_res = a_res

    def reset_for_next_prompt(self) -> None:
        self._prev_gate = None
        self._prev_v_res = None
        self._prev_a_res = None


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-dir", required=True, help="HF repo or local dir, e.g. dgrauet/ltx-2.3-mlx-q8")
    p.add_argument("--prompts", required=True, type=Path, help="Path to a text file, one prompt per line")
    p.add_argument("--num-steps", type=int, default=30, help="Stage 1 denoising steps (default 30)")
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--width", type=int, default=704)
    p.add_argument("--num-frames", type=int, default=97)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cfg-scale", type=float, default=3.0)
    p.add_argument("--stg-scale", type=float, default=0.0)
    p.add_argument("--polyfit-degree", type=int, default=4)
    p.add_argument(
        "--default-rel-l1-thresh",
        type=float,
        default=0.15,
        help="Starting threshold to ship with the preset (user can tune).",
    )
    p.add_argument("--out", type=Path, default=Path("ltx2_teacache_calibration.json"))
    return p.parse_args()


def main() -> int:
    args = _parse_args()

    if not args.prompts.exists():
        print(f"prompts file not found: {args.prompts}", file=sys.stderr)
        return 2

    prompts = [line.strip() for line in args.prompts.read_text().splitlines() if line.strip()]
    if not prompts:
        print("prompts file is empty", file=sys.stderr)
        return 2

    print(f"Calibrating TeaCache for LTX-2 stage 1: {len(prompts)} prompts × {args.num_steps} steps")

    pipeline = TwoStagePipeline(model_dir=args.model_dir)
    calibrator = _StreamingCalibrator()

    for i, prompt in enumerate(prompts):
        print(f"[{i + 1}/{len(prompts)}] {prompt[:80]}…")
        calibrator.reset_for_next_prompt()

        # tap fires on each stage 1 step's conditioned pass. Stage 2 (3 steps)
        # also runs after stage 1; the tap continues firing but those late
        # steps are dropped by the prompt-boundary reset on the next iteration.
        # If you want strictly stage-1-only deltas, run with --num-steps and
        # discard the trailing 3 deltas per prompt.
        pipeline.generate_two_stage(
            prompt=prompt,
            height=args.height,
            width=args.width,
            num_frames=args.num_frames,
            seed=args.seed + i,
            stage1_steps=args.num_steps,
            stage2_steps=3,
            cfg_scale=args.cfg_scale,
            stg_scale=args.stg_scale,
            tap=calibrator,
        )

    if not calibrator.delta_in:
        print("No deltas captured — did the tap fire? See plan follow-up.", file=sys.stderr)
        return 1

    coeffs = np.polyfit(calibrator.delta_in, calibrator.delta_res_video, deg=args.polyfit_degree).tolist()

    out_payload = {
        "name": "ltx2",
        "coefficients": coeffs,
        "rel_l1_thresh": args.default_rel_l1_thresh,
        "calibration_meta": {
            "num_prompts": len(prompts),
            "num_steps": args.num_steps,
            "height": args.height,
            "width": args.width,
            "num_frames": args.num_frames,
            "model_dir": args.model_dir,
            "polyfit_degree": args.polyfit_degree,
            "delta_in_count": len(calibrator.delta_in),
        },
    }
    args.out.write_text(json.dumps(out_payload, indent=2))

    print(f"\n  Wrote {args.out}")
    print("\nAdd this to mlx_arsenal/diffusion/teacache.py TEACACHE_PRESETS:\n")
    print('    "ltx2": {')
    print(f"        \"coefficients\": {coeffs!r},")
    print(f"        \"rel_l1_thresh\": {args.default_rel_l1_thresh},")
    print("    },")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Smoke-import the script (no execution; we only verify it parses and imports)**

Run: `cd /Users/dgrauet/Work/ltx-2-mlx && python3 -c "from ltx_pipelines_mlx.scripts import calibrate_teacache; print(calibrate_teacache.__doc__[:60])"`

Expected: prints the first 60 chars of the module docstring without any ImportError.

- [ ] **Step 5: Run ruff on the new file and modified files**

Run: `cd /Users/dgrauet/Work/ltx-2-mlx && python3 -m ruff check packages/ltx-pipelines-mlx/scripts/calibrate_teacache.py packages/ltx-pipelines-mlx/src/ltx_pipelines_mlx/ti2vid_two_stages.py`

Expected: `All checks passed!`

- [ ] **Step 6: Run the previously-passing tests to confirm no regression**

Run: `cd /Users/dgrauet/Work/ltx-2-mlx && python3 -m pytest tests/test_teacache_integration.py tests/test_two_stage.py -q`

Expected: all PASS.

- [ ] **Step 7: Commit**

```bash
cd /Users/dgrauet/Work/ltx-2-mlx
git add packages/ltx-pipelines-mlx/scripts/calibrate_teacache.py packages/ltx-pipelines-mlx/scripts/__init__.py packages/ltx-pipelines-mlx/src/ltx_pipelines_mlx/ti2vid_two_stages.py
git commit -m "$(cat <<'EOF'
feat: add TeaCache calibration script for LTX-2 stage 1

Streams (gate, video_residual, audio_residual) per step from the
conditioned pass via the new tap kwarg on generate_two_stage. Computes
relative L1 deltas in fp32, fits a degree-4 polynomial, and prints a
copy-pasteable preset snippet for mlx_arsenal TEACACHE_PRESETS["ltx2"].

Memory-friendly: rolling state per stream, no per-step tensor buffering.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Self-Review Checklist (run after all tasks done)

- [ ] All committed; `git status` clean in both repos.
- [ ] `cd /Users/dgrauet/Work/mlx-arsenal && python3 -m pytest tests/ -q` → all green.
- [ ] `cd /Users/dgrauet/Work/ltx-2-mlx && python3 -m pytest tests/ -q` → all green (or only fails are pre-existing real-weight tests that need a model download).
- [ ] `python3 -m ruff check ...` → clean across both repos for files touched.
- [ ] `enable_teacache=True` raises a clear RuntimeError mentioning `calibrate_teacache` (verified by `TestPipelineFlag`).
- [ ] No file outside the spec was modified.

## Open follow-ups (NOT part of this plan)

These are documented in the spec's "Open questions" section — leave them for a follow-up plan after the user runs calibration:

1. Add the `ltx2` preset to `mlx-arsenal` once calibration produces real coefficients.
2. Decide whether the boundary rule (first/last step always compute) is appropriate for stage 1's truncated schedule (verify empirically with one end-to-end run, post-calibration).
3. fp32-cast micro-optimization in `_rel_l1_fp32` (cast before subtract vs. after — measure on a real run).
4. Optional: ComfyUI-LTXVideo-mlx integration once stable.
