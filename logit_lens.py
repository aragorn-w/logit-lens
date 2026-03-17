#!/usr/bin/env python3
"""
Logit Lens Terminal Visualizer

Logit Lens (nostalgebraist, 2020): Projects the residual stream at each
transformer layer through the final LayerNorm and unembedding matrix to
reveal how a language model's predictions evolve across layers.

The residual stream at any intermediate layer can be decoded into a token
distribution by applying the same unembedding used at the final layer,
showing what the model would predict if it stopped computation there.

Uses TransformerLens for model loading and hook-based activation caching,
and Rich for colored terminal output with background-color heatmaps.

Usage:
    uv run python logit_lens.py "The Eiffel Tower is located in the city of"
    uv run python logit_lens.py --model gpt2-medium "Hello world"
    uv run python logit_lens.py --top-k 3 "The capital of France is"
"""

import contextlib
import io
import logging
import math
import warnings

# Suppress noisy deprecation warnings before any library imports
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*torch_dtype.*")
warnings.filterwarnings("ignore", message=".*deprecated.*")
logging.disable(logging.WARNING)

import torch  # noqa: E402
import transformer_lens  # noqa: E402
import typer  # noqa: E402
from rich.console import Console  # noqa: E402
from rich.markup import escape  # noqa: E402
from rich.table import Table  # noqa: E402
from rich.text import Text  # noqa: E402

console = Console(soft_wrap=True)
app = typer.Typer(add_completion=False)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_transformer_lens_model(
    model_name: str,
) -> transformer_lens.HookedTransformer:
    """Load a pretrained model via TransformerLens."""
    console.print(f"[bold]Loading {model_name} (TransformerLens)...[/bold]")
    with (
        contextlib.redirect_stderr(io.StringIO()),
        contextlib.redirect_stdout(io.StringIO()),
    ):
        model = transformer_lens.HookedTransformer.from_pretrained(model_name)
    model.eval()
    console.print(
        f"[green]Loaded {model_name} "
        f"({model.cfg.n_layers} layers, d_model={model.cfg.d_model})[/green]\n"
    )
    return model


# ---------------------------------------------------------------------------
# Logit lens
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_logit_lens(
    model: transformer_lens.HookedTransformer, prompt: str, top_k: int = 1
) -> tuple[list[str], list[list[list[tuple[str, float]]]]]:
    """
    Run the logit lens: project each layer's residual stream through the
    unembedding to get per-layer token predictions.

    Returns:
        input_tokens: list of string tokens from the prompt
        layer_predictions: [n_layers+1][seq_len] -> list of (token_str, prob) top-k
    """
    tokens = model.to_tokens(prompt)  # (1, seq_len)
    _, cache = model.run_with_cache(tokens)

    n_layers = model.cfg.n_layers

    residual_states = []
    residual_states.append(cache["hook_embed"] + cache["hook_pos_embed"])
    for layer in range(n_layers):
        residual_states.append(cache[f"blocks.{layer}.hook_resid_post"])

    layer_predictions: list[list[list[tuple[str, float]]]] = []

    for state in residual_states:
        normed = model.ln_final(state)
        logits = model.unembed(normed)
        probs = torch.softmax(logits[0], dim=-1)
        top_probs, top_indices = probs.topk(top_k, dim=-1)

        pos_preds = []
        for pos in range(probs.shape[0]):
            preds = []
            for k in range(top_k):
                tok_str = model.to_string(top_indices[pos, k].item())
                preds.append((tok_str, top_probs[pos, k].item()))
            pos_preds.append(preds)
        layer_predictions.append(pos_preds)

    input_token_strs = [model.to_string(t.item()) for t in tokens[0]]
    return input_token_strs, layer_predictions


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------

def prob_to_bg_fg(prob: float) -> tuple[str, str]:
    """
    Map probability [0,1] to (background_rgb, foreground) using a smooth
    red-to-green gradient. Font color is black or white for contrast.
    """
    if math.isnan(prob):
        prob = 0.0
    elif math.isinf(prob):
        prob = 1.0 if prob > 0 else 0.0
    p = max(0.0, min(1.0, prob))

    if p < 0.5:
        t = p / 0.5
        r = 200
        g = int(160 * t)
        b = 0
    else:
        t = (p - 0.5) / 0.5
        r = int(200 * (1 - t))
        g = 160 + int(40 * t)
        b = 0

    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    fg = "black" if luminance > 60 else "white"

    bg = f"rgb({r},{g},{b})"
    return bg, fg


def sanitize_token(tok: str) -> str:
    """Make a token string safe and readable for terminal display."""
    s = repr(tok)[1:-1]
    if len(s) > 12:
        s = s[:11] + "\u2026"
    return s


def display_lens(
    input_tokens: list[str],
    layer_predictions: list[list[list[tuple[str, float]]]],
    prompt: str,
    lens_type: str = "Logit",
    has_embed_layer: bool = True,
) -> None:
    """Render the lens output as a colored table in the terminal."""
    seq_len = len(input_tokens)
    n_rows = len(layer_predictions)

    console.print(
        f'[bold underline]{lens_type} Lens: "{escape(prompt)}"[/bold underline]\n'
    )

    table = Table(
        title="Top predicted next token at each layer",
        show_lines=True,
        padding=(0, 1),
        expand=False,
    )

    table.add_column("Layer", style="bold", width=10, no_wrap=True)
    for tok in input_tokens:
        table.add_column(
            sanitize_token(tok), no_wrap=True, min_width=8, max_width=16
        )

    for row_idx in range(n_rows):
        if has_embed_layer:
            # Logit lens: Embed, L0, L1, ..., L{n-1} (out)
            if row_idx == 0:
                label = "Embed"
            elif row_idx == n_rows - 1:
                label = f"L{row_idx - 1} (out)"
            else:
                label = f"L{row_idx - 1}"
        else:
            # Alternative labeling: L0, L1, ..., L{n-1}, Output
            if row_idx == n_rows - 1:
                label = "Output"
            else:
                label = f"L{row_idx}"

        row_cells: list[str | Text] = [label]
        for pos in range(seq_len):
            preds = layer_predictions[row_idx][pos]
            top_prob = preds[0][1]
            bg, fg = prob_to_bg_fg(top_prob)

            lines = []
            for tok_str, _ in preds:
                lines.append(sanitize_token(tok_str))
            cell = Text("\n".join(lines), style=f"{fg} on {bg}")
            row_cells.append(cell)
        table.add_row(*row_cells)

    console.print(table)

    # Smooth gradient legend
    console.print()
    legend = Text("Confidence: ", style="bold")
    legend.append("0%", style="bold")
    for pct in range(0, 101):
        bg, fg = prob_to_bg_fg(pct / 100.0)
        legend.append("\u2588", style=bg)
    legend.append(" 100%", style="bold")
    console.print(legend)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@app.command()
def visualize(
    prompt: str = typer.Argument(
        default="The Eiffel Tower is located in the city of",
        help="Text prompt to analyze.",
    ),
    model: str = typer.Option(
        "gpt2", "--model", "-m",
        help="Model name (e.g. gpt2, gpt2-medium).",
    ),
    top_k: int = typer.Option(
        1, "--top-k", "-k",
        help="Number of top predictions to show per cell.",
        min=1, max=5,
    ),
) -> None:
    """Visualize how predictions evolve across transformer layers."""
    tl_model = load_transformer_lens_model(model)
    input_tokens, layer_predictions = run_logit_lens(
        tl_model, prompt, top_k=top_k
    )
    display_lens(
        input_tokens, layer_predictions, prompt,
        lens_type="Logit", has_embed_layer=True,
    )


if __name__ == "__main__":
    app()
