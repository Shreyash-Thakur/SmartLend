"""Unit tests for the convergence-analysis helpers.

Only the pure logic is tested here — the grid and the plateau verdict — because
they decide what the expensive runs measure and how the headline conclusion is
worded. The model fits themselves are exercised by running the script.
"""

from __future__ import annotations

from research.analysis.convergence import FOLD_STD, convergence_verdict, size_grid


class TestSizeGrid:
    def test_always_ends_at_full_train_size(self):
        assert size_grid(246_008)[-1] == 246_008
        assert size_grid(30_000)[-1] == 30_000

    def test_drops_sizes_at_or_above_n_train(self):
        grid = size_grid(10_000)
        assert grid == [2_000, 5_000, 10_000]

    def test_quick_mode_is_a_prefix_plus_full(self):
        grid = size_grid(246_008, quick=True)
        assert grid == [2_000, 5_000, 10_000, 25_000, 246_008]

    def test_strictly_increasing(self):
        grid = size_grid(246_008)
        assert all(a < b for a, b in zip(grid, grid[1:]))


class TestConvergenceVerdict:
    def test_flat_curve_is_converged(self):
        v = convergence_verdict([10_000, 100_000, 200_000], [0.760, 0.767, 0.7672])
        assert v["converged"] is True
        assert v["half_size"] == 100_000
        assert v["gain_half_to_full"] == round(0.7672 - 0.767, 4)

    def test_rising_curve_is_not_converged(self):
        v = convergence_verdict([10_000, 100_000, 200_000], [0.70, 0.74, 0.76])
        assert v["converged"] is False

    def test_gain_exactly_at_floor_is_not_converged(self):
        # The criterion is strict: gain must be BELOW the noise floor.
        v = convergence_verdict([100, 200], [0.75, 0.75 + FOLD_STD])
        assert v["converged"] is False

    def test_no_half_size_available(self):
        # Grid where every size is > half the full size: verdict must decline,
        # not fabricate a comparison.
        v = convergence_verdict([150, 200], [0.74, 0.75])
        assert v["converged"] is None
