"""Tests for output-dimension snapping (snap_output_dimensions).

Two-stage pipelines generate Stage 1 at half resolution and upscale 2x, so full
dims must be multiples of 64; single-stage pipelines only need multiples of 32.
The helper floors down (never up), clamps to at least one tile, is idempotent,
and logs an informational warning only when a dim actually changes.
"""

import logging

from ltx_core_mlx.components.patchifiers import snap_output_dimensions


class TestTwoStageSnapping:
    def test_default_dims_snap_down(self):
        """The CLI default 480x704 is not a multiple of 64 -> height snaps 480->448."""
        h, w = snap_output_dimensions(480, 704, two_stage=True)
        assert (h, w) == (448, 704)

    def test_both_axes_snap(self):
        # 928 -> 896 (14*64), 480 -> 448 (7*64)
        h, w = snap_output_dimensions(928, 480, two_stage=True)
        assert (h, w) == (896, 448)

    def test_already_divisible_unchanged(self):
        """Negative path: dims already multiples of 64 pass through untouched."""
        for dims in [(64, 64), (960, 960), (512, 768), (448, 704)]:
            assert snap_output_dimensions(*dims, two_stage=True) == dims

    def test_clamps_to_one_tile(self):
        """Below one 64px tile clamps up to 64 rather than 0."""
        assert snap_output_dimensions(10, 10, two_stage=True) == (64, 64)


class TestSingleStageSnapping:
    def test_default_dims_pass_through(self):
        """480x704 is clean for single-stage (both multiples of 32)."""
        assert snap_output_dimensions(480, 704, two_stage=False) == (480, 704)

    def test_snap_to_32(self):
        # 100 -> 96 (3*32)
        assert snap_output_dimensions(100, 100, two_stage=False) == (96, 96)

    def test_between_grids(self):
        """A multiple of 32 but not 64 is fine single-stage, would snap two-stage."""
        assert snap_output_dimensions(480, 480, two_stage=False) == (480, 480)
        assert snap_output_dimensions(480, 480, two_stage=True) == (448, 448)

    def test_clamps_to_one_tile(self):
        assert snap_output_dimensions(5, 5, two_stage=False) == (32, 32)


class TestIdempotency:
    def test_two_stage_idempotent(self):
        once = snap_output_dimensions(928, 480, two_stage=True)
        twice = snap_output_dimensions(*once, two_stage=True)
        assert once == twice

    def test_single_stage_idempotent(self):
        once = snap_output_dimensions(100, 100, two_stage=False)
        twice = snap_output_dimensions(*once, two_stage=False)
        assert once == twice


class TestWarning:
    def test_warns_only_when_changed(self, caplog):
        with caplog.at_level(logging.WARNING, logger="ltx_core_mlx.components.patchifiers"):
            snap_output_dimensions(480, 704, two_stage=True)  # snaps
        assert len(caplog.records) == 1

        caplog.clear()
        with caplog.at_level(logging.WARNING, logger="ltx_core_mlx.components.patchifiers"):
            snap_output_dimensions(448, 704, two_stage=True)  # no change
        assert caplog.records == []

    def test_warn_disabled(self, caplog):
        with caplog.at_level(logging.WARNING, logger="ltx_core_mlx.components.patchifiers"):
            snap_output_dimensions(480, 704, two_stage=True, warn=False)
        assert caplog.records == []

    def test_two_stage_message_content(self, caplog):
        with caplog.at_level(logging.WARNING, logger="ltx_core_mlx.components.patchifiers"):
            snap_output_dimensions(480, 704, two_stage=True)
        msg = caplog.records[0].getMessage()
        assert "two-stage" in msg
        assert "multiples of 64" in msg
        assert "448x704" in msg  # snapped output
        assert "requested 480x704" in msg
        assert "--single-stage" in msg  # actionable hint

    def test_single_stage_message_has_no_single_stage_hint(self, caplog):
        with caplog.at_level(logging.WARNING, logger="ltx_core_mlx.components.patchifiers"):
            snap_output_dimensions(100, 100, two_stage=False)
        msg = caplog.records[0].getMessage()
        assert "single-stage" in msg
        assert "multiples of 32" in msg
        assert "96x96" in msg
        assert "--single-stage" not in msg  # no self-referential hint
