"""Text generation with multiple decoding strategies.

Terminology:
    Autoregressive generation: Produce one token at a time, feeding each output back
        as input for the next step. This is how GPT, Llama, Claude all generate text.

    Temperature: Scale applied to logits before softmax. Controls randomness.
        - temperature=0: Greedy (always pick most likely token). Deterministic but repetitive.
        - temperature=0.7-0.9: Balanced (good for most use cases).
        - temperature=1.0: Sample from the model's raw distribution.
        - temperature>1.0: More random / creative / chaotic.
        Mathematically: softmax(logits / temperature). Higher T → flatter distribution.

    Top-k sampling (Fan et al., 2018): Only consider the k most likely tokens.
        Eliminates low-probability "tail" tokens that can derail generation.
        Typical values: k=40-100.

    Top-p / Nucleus sampling (Holtzman et al., 2020): Only consider the smallest set
        of tokens whose cumulative probability exceeds p. Adapts to the distribution —
        when the model is confident, fewer tokens are considered; when uncertain, more.
        Typical values: p=0.9-0.95.

    Repetition penalty (Keskar et al., 2019): Reduce the probability of tokens that
        have already appeared in the generated text. Prevents degenerate repetition loops.

    Greedy decoding: Always pick the most probable token. Fast but often produces
        bland, repetitive text. Good for code; bad for creative writing.

    Sampling: Randomly draw from the probability distribution. Produces diverse text
        but can be incoherent if temperature is too high.
"""

import torch
import torch.nn.functional as F


@torch.no_grad()
def generate(model, prompt_ids, max_new_tokens, tokenizer=None,
             temperature=0.8, top_k=40, top_p=0.9,
             repetition_penalty=1.0, stop_token_id=None,
             stream=False):
    """Generate text autoregressively with configurable decoding.

    Args:
        model: GPT model (in eval mode)
        prompt_ids: (1, seq_len) tensor of prompt token IDs
        max_new_tokens: maximum tokens to generate
        tokenizer: optional, for streaming output
        temperature: randomness control (0 = greedy, higher = more random)
        top_k: only sample from top k tokens (None = no filtering)
        top_p: nucleus sampling threshold (None = no filtering)
        repetition_penalty: penalize repeated tokens (1.0 = no penalty)
        stop_token_id: stop generation when this token is produced
        stream: if True and tokenizer provided, print tokens as they're generated

    Returns:
        full_ids: (1, prompt_len + generated_len) tensor
    """
    model.eval()
    device = next(model.parameters()).device
    idx = prompt_ids.to(device)
    block_size = model.config.block_size

    generated_ids = []

    for i in range(max_new_tokens):
        # Crop context to block_size
        idx_cond = idx if idx.size(1) <= block_size else idx[:, -block_size:]

        # Forward pass — get logits for last position
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :]  # (1, vocab_size)

        # --- Repetition penalty ---
        if repetition_penalty != 1.0:
            logits = _apply_repetition_penalty(logits, idx, repetition_penalty)

        # --- Temperature ---
        if temperature == 0:
            # Greedy: pick most likely
            idx_next = logits.argmax(dim=-1, keepdim=True)
        else:
            logits = logits / temperature

            # --- Top-k filtering ---
            if top_k is not None:
                logits = _top_k_filter(logits, top_k)

            # --- Top-p (nucleus) filtering ---
            if top_p is not None:
                logits = _top_p_filter(logits, top_p)

            # Sample
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)

        # Append
        idx = torch.cat([idx, idx_next], dim=1)
        generated_ids.append(idx_next.item())

        # Stream output
        if stream and tokenizer:
            token_text = tokenizer.decode([idx_next.item()])
            print(token_text, end="", flush=True)

        # Stop on end token
        if stop_token_id is not None and idx_next.item() == stop_token_id:
            break

    if stream:
        print()  # newline after streaming

    return idx


def _apply_repetition_penalty(logits, past_ids, penalty):
    """Reduce logits for tokens that already appeared in the sequence.

    For each token in past_ids:
        if logit > 0: logit /= penalty  (reduce positive logits)
        if logit < 0: logit *= penalty  (make negative logits more negative)
    """
    for token_id in past_ids[0].unique():
        if logits[0, token_id] > 0:
            logits[0, token_id] /= penalty
        else:
            logits[0, token_id] *= penalty
    return logits


def _top_k_filter(logits, k):
    """Keep only the top k tokens, set rest to -inf."""
    k = min(k, logits.size(-1))
    top_k_values, _ = torch.topk(logits, k)
    threshold = top_k_values[:, -1:]  # kth largest value
    logits = logits.masked_fill(logits < threshold, float('-inf'))
    return logits


def _top_p_filter(logits, p):
    """Nucleus sampling: keep smallest set of tokens with cumulative prob >= p."""
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

    # Find cutoff: first position where cumulative prob exceeds p
    sorted_mask = cumulative_probs - F.softmax(sorted_logits, dim=-1) >= p
    sorted_logits[sorted_mask] = float('-inf')

    # Scatter back to original order
    logits = sorted_logits.scatter(1, sorted_indices, sorted_logits)
    return logits
