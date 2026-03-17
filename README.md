# Logit Lens

Terminal visualizer for the Logit Lens (nostalgebraist, 2020), applied to GPT-2 with TransformerLens.

## File Map

| File | Description |
|------|-------------|
| `logit_lens.py` | Core implementation: model loading, logit lens computation (`run_logit_lens`), color-mapped terminal visualization (`display_lens`), CLI via typer |
| `test_logit_lens.py` | Test suite (47 tests): color mapping, token sanitization, GPT-2 integration, rendering |
| `pyproject.toml` | Dependencies and project metadata (managed with uv) |

## Overview

The "logit lens" is a trick for interpretability from a 2020 blog post by nostalgebraist. It's a fairly intuitive concept. In a transformer, there is a residual stream in each layer, which is in the same space as the final output. So, in fact, the unembedding matrix `W_U`, which is only ever applied at the very end, can be applied to *any* layer to see what the model would predict if it stopped there. See [here for the Tuned Lens repo](https://github.com/aragorn-w/tuned-lens).

In other words, early layers will mostly just predict generic/common tokens, since they haven't been "tuned" by attention and MLPs yet, and later layers will slowly converge to what the actual answer is. It's a no-cost technique, no training or parameters required.

## How It Works

At each layer `l`:

```
logits_l = W_U * LN(h_l)
```

where `h_l` is the residual stream at layer `l`, `LN` is the final LayerNorm, and `W_U` is the unembedding matrix. Softmax gives probabilities, and the top-k tokens are displayed.

The first row (Embed) shows predictions from just the token + position embeddings, before any attention or MLP computation. Each subsequent row adds one more transformer block. The final row matches the model's actual output.

## Installation

Clone the repository and sync dependencies:

```bash
git clone https://github.com/aragorn-w/logit-lens
cd logit-lens
uv sync
```

To install development dependencies (for running tests):

```bash
uv sync --group dev
```

## Usage

Run the logit lens on a prompt:

```bash
uv run python logit_lens.py "The Eiffel Tower is located in the city of"
```

Use a different model:

```bash
uv run python logit_lens.py --model gpt2-medium "The capital of France is"
```

Show top-3 predictions per cell:

```bash
uv run python logit_lens.py --top-k 3 "Hello world"
```

The script will download the model on first run (cached for subsequent runs), then display a colored table in the terminal.

## Example Output

The output is a Rich table where each row is a transformer layer and each column is an input token position. Each cell shows the model's top predicted next token at that layer and position. Cell background colors indicate confidence:

- Red background: low probability (the model is uncertain)
- Yellow background: moderate probability
- Green background: high probability (the model is confident)

A gradient legend is provided below the table for reference.

For a prompt like "The Eiffel Tower is located in the city of", the early layers will be predicting common tokens, and then "Paris" appears around the 8-9th layers and stays locked in until the final output.

## What I Found

Running the logit lens on GPT-2 with various prompts, a few things stood out:

- **Factual recall has a sharp transition.** For a prompt like "The Eiffel Tower is located in the city of," the answer ("Paris") doesn't gradually increase in probability but is instead absent until around layer 8 or 9, at which point it's present and stays that way. It looks like factual recall is "looked up" at a certain depth.

- **The early layers are mostly noise.** The embedding layer and the first few transformer blocks are just making guesses at the most common words in the vocabulary, like "the," "of," and "and," and are not influenced by the prompt at this stage. Again, this makes sense, since these layers haven't had a chance to see the prompt multiple times.

- **Syntactic predictions are made before semantic ones.** For a prompt like "The cat sat on the," the model makes guesses at the kind of word that should come next before making a guess at the actual word.

## Why This Is Useful

The logit lens is essentially free, i.e., it requires no learning, no parameters to be learned, and works for any model with a residual stream. It provides a quick check to make sure the model is "on track" at various layers and can be a great first tool before going to heavier tools. It can also be a great teaching tool to understand how a transformer works layer-wise to make a prediction.

## Problems and Limitations

The main problem is that **intermediate representations were not trained to be compatible with the final unembedding**. When we apply the logit lens, we apply `W_U` and the final `LayerNorm` to representations that were never optimized for this transformation. This leads to:

- Noisy and nonsensical predictions in early layers are not necessarily because those layers are uninformative, but because this information is simply not in a form that the unembedding can decode.
- The heatmap for confidence is also misleading because a red cell does not necessarily mean that this layer has not figured anything out – it just means that what it does know is not compatible with this output space.

The tuned lens (Belrose et al., 2023) addresses this directly by learning per-layer affine probes, which is what the [companion repo](https://github.com/aragorn-w/tuned-lens) implements.

## Running Tests

```bash
uv run pytest
```

Unit tests for color mapping and token sanitization, integration tests that load GPT-2 and verify output shapes, and rendering tests for the terminal display.

## References

- nostalgebraist. "interpreting GPT: the logit lens." LessWrong, 2020. https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens
- Neel Nanda et al. "TransformerLens." https://github.com/TransformerLensOrg/TransformerLens
