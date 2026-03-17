"""Tests for logit_lens.py - logit lens visualizer."""

import math
import warnings

warnings.filterwarnings("ignore")

import logging

logging.disable(logging.WARNING)

import contextlib
import io
import re

import torch
import pytest

from logit_lens import (
    load_transformer_lens_model,
    run_logit_lens,
    prob_to_bg_fg,
    sanitize_token,
    display_lens,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_rgb(bg: str) -> tuple[int, int, int]:
    """Parse 'rgb(r,g,b)' -> (r, g, b)."""
    parts = bg[4:-1].split(",")
    return int(parts[0]), int(parts[1]), int(parts[2])


def _make_console_and_buf():
    """Create a Rich Console that writes to a StringIO buffer."""
    buf = io.StringIO()
    from rich.console import Console
    return Console(file=buf, force_terminal=True, width=200), buf


_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]|\x1b\[\?[0-9]*[a-zA-Z]")


def _strip_ansi(text: str) -> str:
    """Remove all ANSI escape sequences from a string."""
    return _ANSI_RE.sub("", text)


def _with_test_console(fn, *args, **kwargs):
    """Run fn while the module-level console is swapped out."""
    import logit_lens
    test_console, buf = _make_console_and_buf()
    orig = logit_lens.console
    logit_lens.console = test_console
    try:
        fn(*args, **kwargs)
    finally:
        logit_lens.console = orig
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def tl_model():
    """Load TransformerLens GPT-2 once for all tests."""
    with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(
        io.StringIO()
    ):
        import transformer_lens
        model = transformer_lens.HookedTransformer.from_pretrained("gpt2")
    model.eval()
    return model


# ===================================================================
# prob_to_bg_fg -- unit tests
# ===================================================================

class TestProbToBgFg:
    def test_returns_tuple_of_two_strings(self):
        bg, fg = prob_to_bg_fg(0.5)
        assert isinstance(bg, str)
        assert isinstance(fg, str)

    def test_zero_prob_is_red(self):
        bg, _ = prob_to_bg_fg(0.0)
        r, g, b = _parse_rgb(bg)
        assert r > 100
        assert g == 0

    def test_full_prob_is_green(self):
        bg, _ = prob_to_bg_fg(1.0)
        r, g, b = _parse_rgb(bg)
        assert r == 0
        assert g >= 150

    def test_mid_prob_is_yellowish(self):
        bg, _ = prob_to_bg_fg(0.5)
        r, g, b = _parse_rgb(bg)
        assert r > 50
        assert g > 50

    def test_gradient_green_monotonic_increasing(self):
        prev_g = -1
        for pct in range(0, 101):
            bg, _ = prob_to_bg_fg(pct / 100)
            g = _parse_rgb(bg)[1]
            assert g >= prev_g, (
                f"Green not monotonic at {pct}%: got {g}, prev {prev_g}"
            )
            prev_g = g

    def test_gradient_red_monotonic_decreasing(self):
        prev_r = 999
        for pct in range(0, 101):
            bg, _ = prob_to_bg_fg(pct / 100)
            r = _parse_rgb(bg)[0]
            assert r <= prev_r, (
                f"Red not monotonic at {pct}%: got {r}, prev {prev_r}"
            )
            prev_r = r

    def test_fg_is_black_or_white(self):
        for pct in range(0, 101):
            _, fg = prob_to_bg_fg(pct / 100)
            assert fg in ("black", "white")

    def test_clamps_negative(self):
        assert prob_to_bg_fg(-0.5) == prob_to_bg_fg(0.0)

    def test_clamps_above_one(self):
        assert prob_to_bg_fg(1.5) == prob_to_bg_fg(1.0)

    def test_dark_bg_gets_white_fg(self):
        _, fg = prob_to_bg_fg(0.0)
        assert fg == "white"

    def test_bright_bg_gets_black_fg(self):
        _, fg = prob_to_bg_fg(0.5)
        assert fg == "black"

    # --- Edge cases ---

    def test_nan_does_not_crash(self):
        bg, fg = prob_to_bg_fg(float("nan"))
        assert bg.startswith("rgb(")
        assert fg in ("black", "white")

    def test_nan_treated_as_zero(self):
        assert prob_to_bg_fg(float("nan")) == prob_to_bg_fg(0.0)

    def test_positive_inf_treated_as_one(self):
        assert prob_to_bg_fg(float("inf")) == prob_to_bg_fg(1.0)

    def test_negative_inf_treated_as_zero(self):
        assert prob_to_bg_fg(float("-inf")) == prob_to_bg_fg(0.0)

    def test_boundary_continuity_at_half(self):
        """No large discontinuity at the p=0.5 boundary between the two branches."""
        bg_below, _ = prob_to_bg_fg(0.499)
        bg_at, _ = prob_to_bg_fg(0.5)
        bg_above, _ = prob_to_bg_fg(0.501)

        r1, g1, _ = _parse_rgb(bg_below)
        r2, g2, _ = _parse_rgb(bg_at)
        r3, g3, _ = _parse_rgb(bg_above)

        # Red channel: should be ~200 on both sides of boundary
        assert abs(r1 - r2) <= 5
        # Green: should be ~160 on both sides
        assert abs(g1 - g2) <= 2
        assert abs(g2 - g3) <= 2

    def test_blue_is_always_zero(self):
        for pct in range(0, 101, 5):
            bg, _ = prob_to_bg_fg(pct / 100)
            _, _, b = _parse_rgb(bg)
            assert b == 0

    def test_rgb_values_in_valid_range(self):
        for pct in range(0, 101):
            bg, _ = prob_to_bg_fg(pct / 100)
            r, g, b = _parse_rgb(bg)
            assert 0 <= r <= 255
            assert 0 <= g <= 255
            assert 0 <= b <= 255


# ===================================================================
# sanitize_token -- unit tests
# ===================================================================

class TestSanitizeToken:
    def test_normal_word(self):
        assert sanitize_token("hello") == "hello"

    def test_leading_space(self):
        result = sanitize_token(" Tower")
        assert "Tower" in result

    def test_newline_escaped(self):
        assert "\\n" in sanitize_token("\n")

    def test_tab_escaped(self):
        assert "\\t" in sanitize_token("\t")

    def test_carriage_return_escaped(self):
        assert "\\r" in sanitize_token("\r")

    def test_null_byte_escaped(self):
        result = sanitize_token("\x00")
        assert "\x00" not in result  # should be escaped

    def test_truncates_long_token(self):
        result = sanitize_token("a" * 50)
        assert len(result) == 12
        assert result.endswith("\u2026")

    def test_short_token_not_truncated(self):
        assert sanitize_token("hi") == "hi"

    def test_empty_string(self):
        result = sanitize_token("")
        assert result == ""

    def test_exactly_12_chars_not_truncated(self):
        tok = "a" * 12
        result = sanitize_token(tok)
        assert result == tok
        assert "\u2026" not in result

    def test_exactly_13_chars_truncated(self):
        tok = "a" * 13
        result = sanitize_token(tok)
        assert len(result) == 12
        assert result.endswith("\u2026")

    def test_unicode_chars(self):
        result = sanitize_token("cafe\u0301")
        assert "caf" in result

    def test_backslash(self):
        result = sanitize_token("a\\b")
        # repr will escape the backslash
        assert "\\\\" in result


# ===================================================================
# run_logit_lens -- integration tests
# ===================================================================

class TestRunLogitLens:
    def test_basic_output_shape(self, tl_model):
        input_tokens, layer_preds = run_logit_lens(tl_model, "Hello", top_k=1)
        assert len(input_tokens) >= 1
        assert len(layer_preds) == tl_model.cfg.n_layers + 1
        for row in layer_preds:
            assert len(row) == len(input_tokens)

    def test_top_k_predictions(self, tl_model):
        _, layer_preds = run_logit_lens(tl_model, "Hello", top_k=3)
        for row in layer_preds:
            for cell in row:
                assert len(cell) == 3
                for tok_str, prob in cell:
                    assert isinstance(tok_str, str)
                    assert 0.0 <= prob <= 1.0

    def test_probs_are_valid(self, tl_model):
        _, layer_preds = run_logit_lens(tl_model, "The cat sat on", top_k=1)
        for row in layer_preds:
            for cell in row:
                _, prob = cell[0]
                assert 0.0 <= prob <= 1.0
                assert not math.isnan(prob)

    def test_final_layer_matches_model_output(self, tl_model):
        prompt = "The cat"
        _, layer_preds = run_logit_lens(tl_model, prompt, top_k=1)
        tokens = tl_model.to_tokens(prompt)
        logits = tl_model(tokens)
        probs = torch.softmax(logits[0], dim=-1)
        _, top_idx = probs.max(dim=-1)
        model_pred = tl_model.to_string(top_idx[-1].item())
        lens_pred = layer_preds[-1][-1][0][0]
        assert lens_pred == model_pred

    def test_different_prompts_differ(self, tl_model):
        _, preds_a = run_logit_lens(tl_model, "The dog", top_k=1)
        _, preds_b = run_logit_lens(tl_model, "import os", top_k=1)
        assert preds_a[-1][-1][0][0] != preds_b[-1][-1][0][0]

    def test_single_token_prompt(self, tl_model):
        """Model prepends BOS, so even a single char gives 2 tokens."""
        input_tokens, layer_preds = run_logit_lens(tl_model, "A", top_k=1)
        assert len(input_tokens) >= 1
        for row in layer_preds:
            assert len(row) == len(input_tokens)

    def test_top_k_probs_are_sorted_descending(self, tl_model):
        _, layer_preds = run_logit_lens(tl_model, "The", top_k=5)
        for row in layer_preds:
            for cell in row:
                probs = [p for _, p in cell]
                assert probs == sorted(probs, reverse=True), (
                    f"Top-k probs should be descending: {probs}"
                )

    def test_top_k_1_equals_top_k_5_first(self, tl_model):
        """Top-1 prediction should match the first of top-5."""
        _, preds1 = run_logit_lens(tl_model, "Hello world", top_k=1)
        _, preds5 = run_logit_lens(tl_model, "Hello world", top_k=5)
        for row1, row5 in zip(preds1, preds5):
            for cell1, cell5 in zip(row1, row5):
                assert cell1[0][0] == cell5[0][0]
                assert abs(cell1[0][1] - cell5[0][1]) < 1e-5


# ===================================================================
# display_lens -- rendering tests
# ===================================================================

class TestDisplayLens:
    def test_display_logit_lens_runs(self, tl_model):
        input_tokens, layer_preds = run_logit_lens(tl_model, "Hi", top_k=1)
        output = _with_test_console(
            display_lens, input_tokens, layer_preds, "Hi",
            lens_type="Logit", has_embed_layer=True,
        )
        assert "Logit Lens" in output
        assert "Embed" in output
        assert "Confidence" in output

    def test_no_percentages_in_table_cells(self, tl_model):
        input_tokens, layer_preds = run_logit_lens(tl_model, "Hi", top_k=1)
        output = _with_test_console(
            display_lens, input_tokens, layer_preds, "Hi",
            lens_type="Logit", has_embed_layer=True,
        )
        table_part = output.split("Confidence")[0]
        for line in table_part.strip().split("\n"):
            if "Layer" in line or "\u2501" in line or "\u2500" in line:
                continue
            matches = re.findall(r"\d{1,3}%", line)
            assert len(matches) == 0, (
                f"Found percentage in table data: {matches} in: {line}"
            )

    def test_top_k_data_structure(self, tl_model):
        _, layer_preds = run_logit_lens(tl_model, "Hi", top_k=2)
        for row in layer_preds:
            for cell in row:
                assert len(cell) == 2

    def test_label_embed_when_single_layer_embed(self):
        """Edge case: 1 row with has_embed_layer -> label should be 'Embed'."""
        fake_preds = [[[("tok", 0.5)]]]
        output = _with_test_console(
            display_lens, ["A"], fake_preds, "A",
            lens_type="Logit", has_embed_layer=True,
        )
        assert "Embed" in output

    def test_logit_label_two_rows(self):
        """Embed + 1 layer -> labels: 'Embed', 'L0 (out)'."""
        fake_preds = [
            [[("emb", 0.1)]],
            [[("out", 0.9)]],
        ]
        output = _with_test_console(
            display_lens, ["X"], fake_preds, "X",
            lens_type="Logit", has_embed_layer=True,
        )
        assert "Embed" in output
        assert "L0 (out)" in output

    def test_display_with_empty_token(self):
        """Display should not crash with an empty token string."""
        fake_preds = [[[("", 0.3)]]]
        # Should not raise
        _with_test_console(
            display_lens, ["test"], fake_preds, "test",
            lens_type="Logit", has_embed_layer=True,
        )

    def test_display_prompt_with_rich_markup_chars(self):
        """Prompt containing [ ] shouldn't break Rich rendering."""
        fake_preds = [[[("a", 0.5)]]]
        output = _with_test_console(
            display_lens, ["["], fake_preds, "test [bracket]",
            lens_type="Logit", has_embed_layer=True,
        )
        # Should not crash and should contain the lens name
        assert "Logit Lens" in output


# ===================================================================
# Model loading -- integration tests
# ===================================================================

class TestModelLoading:
    def test_load_transformer_lens(self):
        import logit_lens
        test_console, buf = _make_console_and_buf()
        orig = logit_lens.console
        logit_lens.console = test_console
        try:
            model = load_transformer_lens_model("gpt2")
        finally:
            logit_lens.console = orig
        assert model.cfg.n_layers == 12
        assert model.cfg.d_model == 768
