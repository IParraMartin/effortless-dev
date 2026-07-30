# Method

Two mechanisms live in this repository and they are deliberately separable.
Sections 1–4 describe the shared backbone and its multi-exit training.
Sections 5–8 describe **request-level vertical routing**, which is the primary
method. Sections 9–13 describe **token-level early exit**, which came first and
is retained as an extension.

The single most important structural difference: request-level routing has *no
cache approximation*. Everything from section 10 onward exists because the
token-level formulation does.

## 1. Backbone

A standard modern decoder-only stack, implemented from scratch: pre-norm
RMSNorm, rotary position embeddings, grouped-query attention through
`F.scaled_dot_product_attention`, SwiGLU feed-forward, no biases, tied
input/output embeddings. Nothing here is unusual; it exists so the early-exit
work sits on a realistic base.

## 2. Exit modules

Every layer in `config.exit_layers` carries an `ExitModule`: a per-layer
`RMSNorm` followed by the model's **shared** output projection, itself tied to
the input embedding.

The sharing is not an optimization detail, it is what makes the idea affordable.
Independent heads would cost `n_exits × d_model × vocab_size` parameters — for a
124M model with 12 exits, an extra **463M**, several times the backbone. What
genuinely has to be per-layer is the normalization: each depth has its own scale
and its own notion of a finished residual stream.

## 3. Confidence

All three criteria return **uncertainty in [0, 1]**, and a token exits when
`uncertainty < threshold`:

| Criterion | Definition | Reads |
|---|---|---|
| `entropy` | Shannon entropy ÷ `log(vocab_size)` | the whole distribution |
| `max_prob` | `1 − max p` | only the top token |
| `top2_margin` | `1 − (p₁ − p₂)` | the two-way ties greedy decoding cares about |

Sharing the direction matters. Entropy is naturally *low* when confident while
the other two are naturally *high*; flipping the comparison per criterion is a
reliable source of silent bugs. Normalizing entropy by `log V` also makes a
threshold portable across tokenizers.

## 4. Training objective

The full stack always runs during training — an exit cannot learn to predict
from a depth it never sees. Early exiting is a generation-time behaviour.

```
total = Σ_i  w_i · [ CE_i + λ · T² · KL(final ‖ exit_i) ]
```

- `w_i` rises linearly with depth, normalized to sum to 1.
- The KL term is self-distillation from the **detached** final layer. Without
  it shallow exits stay weak, rarely clear the threshold, and the whole
  mechanism idles.
- The teacher gets cross-entropy only; it is its own teacher.

**Memory is the binding constraint, not compute.** Logits at every exit are
`n_exits × batch × seq_len × vocab_size`, and `cross_entropy` retains its
log-softmax for backward, so they cannot be freed as the loop advances. Twelve
exits over 2×1024 tokens with a 50k vocabulary is ≈5 GB. `exits_per_step` scores
a rotating subset instead.

Two details in that rotation that matter under DDP:

- The subset is chosen by a **deterministic rotation keyed on a step counter**,
  not a random draw, so every rank selects the same exits without
  communicating. Independent sampling would silently diverge gradients.
- Unselected exits receive no gradient, so DDP requires
  `find_unused_parameters=True`. Without it, training fails with
  *"Expected to have finished reduction in the prior iteration"*.

When a subset is scored, the total shallow weight is **redistributed** across
the exits actually chosen. Scaling by the plain count ratio is also unbiased in
expectation, but the per-exit weights differ sharply by depth, so the per-step
objective would wander (measured: ±25%) and drag gradient clipping and the LR
schedule with it. Redistribution holds it fixed — measured agreement with the
full objective on a fixed batch is 6.245–6.247 versus 6.246.

## 5. Depth-capped execution

Everything request-level is built from one primitive: run blocks `start .. stop`
over an activation, and nothing else.

**Depth means executed blocks**, from 1 to `n_layers`; a layer *index* is one
less. The two conversions live in exactly two functions and nowhere else. This
is not fastidiousness — an off-by-one here already produced a bug where deeper
layers rotated positions at *n+1*, and reported "mean exit layer 4.46 of 5" was
being read as a depth when it was an index.

Three operations:

- `forward_to_depth(ids, d, cache)` runs the first `d` blocks. With a
  depth-capped cache, nothing above `d` is executed *or allocated*.
- `continue_from_depth(hidden, d, D, cache)` runs the suffix over an activation
  the caller already holds.
- `endpoint_logits(hidden, d)` applies that depth's exit normalization and the
  shared output projection, **once**.

### The vocabulary head is the reason the controller looks at hidden states

One projection costs `d_model × vocab_size` multiply-accumulates. At the
repository's 768-wide default with a 52k vocabulary that is 39.9M, against 7.1M
for a whole block excluding attention: **one head is 5.6 blocks**. Testing three
candidate depths by reading the softmax at each would cost more than two blocks
of depth. So the endpoint readout happens once per generated token, at the
chosen depth only, and the counters that prove it are instrumented at a single
call site so they cannot drift from what was computed.

## 6. The depth-capped cache

A request assigned depth `d` allocates entries for layers `0 .. d-1` and
nothing else. Writing above the cap **raises**.

That the cap is enforced rather than merely respected is the point. A cache
that quietly materializes upper layers looks identical from the outside, and
the memory claim — `1 − d/L` of the bytes, exactly, since per-layer width is
uniform — would then be false while every test still passed.

The full-depth cache is unchanged and is what the token-level path uses.

## 7. The prompt probe and the controller

The decision is *not* "is the current answer confident?" but "is more depth
worth buying?". Those come apart. Two prompts can produce identical output
entropy while deeper computation fixes one and changes nothing for the other; a
policy seeing only entropy must treat them alike, so it cannot be optimal. The
quantity to estimate is

```
continue at checkpoint k  <=>  E[V_{k+1} − R_k | H_k] > λ · ΔC_k
```

`DepthController` estimates a practical form of it from pooled probe features:
last real token, masked mean, or both, plus normalized prompt length. It never
touches the vocabulary distribution — for the arithmetic reason in section 5.
It costs `d_model × hidden + hidden × tiers`, three orders of magnitude below
one head call.

Two output heads, answering different questions:

| Head | Predicts | Property |
|---|---|---|
| `utility` | quality at every tier | cost applied at selection, so **λ can change at inference on a frozen controller** — one checkpoint serves a whole frontier |
| `ordinal` | `P(D* ≤ tier_k)` | monotone by construction via cumulative softplus cutpoints, matching the nested structure of the events; easier to calibrate, tied to one operating point |

The cost side is **supplied, not predicted**. Depth cost is deterministic given
the architecture and the prompt, so predicting it would add variance for
nothing, and keeping it external is exactly what lets λ move.

### Padding is handled, not ignored

With right padding and causal attention the *last real* token's state is exact
whatever follows it — but a mean over the padded length is not. It averages in
states that attended to nothing and shrinks by a factor depending on how much
padding a row happened to receive. Both reductions read the real lengths.

## 8. Routed generation

Probe the prompt, pool, choose a depth, finish the prefill only to that depth,
decode there.

Requests are grouped by prompt length and then by chosen depth, and each group
runs separately. **That is what makes the result exact**: a single padded batch
would let one row's padding enter another's attention, and one cache cannot
hold different depths for different rows. Under greedy decoding the tokens are
identical to routing each request alone — verified, with scripted mixed depths
rather than whatever an untrained controller happens to prefer.

This is not a throughput claim. Grouping in Python serializes what a server
would overlap. What it does establish is that the depth cap is real: there are
no upper-layer entries to count, so the memory saving is measured rather than
asserted.

### Escalation

A request can be raised to a deeper endpoint, exactly, provided the boundary
activation was retained **for every prompt position** — not just the last. Upper
blocks need keys and values everywhere they will attend. The strategy is retain
and replay the suffix; the alternative, recomputing the lower prefix, costs
more. The replay is charged as `backfill_tokens` and `backfill_blocks` rather
than described as free, because a reusable prefix is the vertical cascade's
whole cost advantage over a horizontal one and inflating it would be
self-serving.

## 9. What request-level routing gives up, and what it gets

| | Request-level | Token-level |
|---|---|---|
| Granularity | one depth per request | one per token |
| Cache approximation | **none** | propagated states |
| K/V memory saved | `1 − d/L` exactly | none |
| Latency at batch > 1 | yes, if bucketed | no |
| Decision cost | one probe, one small MLP | a softmax per checkpoint |

The rest of this document describes the token-level path.

## 10. The KV-cache problem

This is the constraint that shapes the token-level formulation, and the one
request-level routing does not have.

If token *t* exits at layer 3, layers 4–11 never computed its keys and values.
But token *t+1* may go deeper, and at layer 7 it must attend to token *t*'s
layer-7 key/value — which does not exist. **There is no exact fix.**

### Plain propagation (CALM)

Copy the exit hidden state `h_L` upward and compute only `k_proj`/`v_proj` at
each remaining layer, skipping attention and the FFN. That is where the cost is;
the projections are cheap, especially under GQA.

One detail is easy to get wrong: the propagated state must pass through **that
layer's own `attn_norm`** before projection, because that is what the real
forward feeds to `k_proj`. Skipping it writes plausible-looking but wrong cache
entries and degrades generation without ever raising.

### Batched decoding

A layer runs for the whole batch whenever any row still needs it, so rows that
already exited would otherwise pick up keys and values they were never supposed
to have. Those entries are overwritten with propagated ones
(`KVCache.overwrite_last`), which keeps a row's cache **identical to what it
would be if decoded alone** — verified exactly.

Being honest about what this does and does not buy: correctness is
batch-independent, but *wall-clock savings* only appear once every row in the
batch has exited, or at batch size one. The GPU runs the block either way. This
is the standard caveat and the main obstacle to early exit being deployable.

## 11. Learned KV propagation

Plain propagation bets that residual connections keep adjacent layers similar.
The bet worsens with distance, and nothing in training ever tells the model to
make it hold. `KVPropagator` learns the correction instead: a bottleneck adapter
per layer that predicts the residual stream the target layer would genuinely
have received.

Three properties are deliberate:

- **Zero-initialized output.** An untrained adapter reproduces plain propagation
  *exactly* (verified bit-for-bit), making this a strict generalization of CALM
  rather than a different starting point.
- **Conditioned on the depth gap.** Carrying a state one layer is a different
  problem from carrying it eight.
- **Gap 0 is a hard identity.** When a token exits at layer `L`, layer `L+1`'s
  input *is* the exit state — exact by construction. This is masked, not
  assumed: the training objective never supervises gap 0, so an unmasked adapter
  applies an arbitrary learned offset to a state that was already perfect. Left
  unenforced it measurably corrupted an exact layer (0.000 → 0.142).

### Supervising the right quantity

The adapter is trained against `rms_direction(h_true)` — the RMS-normalized
state — not the raw hidden state.

This was not the first design, and the correction is the more useful finding.
Matching raw hidden states halved the training objective and moved real cache
error **not at all** (0.4829 vs 0.4824). `k_proj` never sees the raw state; it
sees it after RMSNorm, which discards magnitude — and magnitude was most of what
was being minimized. Supervising the post-norm direction targets exactly what
the projection consumes, and `rms_direction` is parameter-free, so gradients
reach only the adapter.

## 12. Exposure-matched fitting

An adapter fitted on full-depth states is solving a cleaner problem than the one
it faces. At inference, every layer beneath the exit attended over a cache
already full of approximations.

Closing that gap normally means decoding token-by-token inside the training
loop, which is far too slow. It is also unnecessary. **The corrupted forward can
be evaluated exactly in one parallel pass.** Attention is strictly causal and a
position's cached keys and values depend only on its own trajectory, so once
each position's exit layer is fixed, every layer can mix true K/V (positions
still alive) with propagated K/V (positions already stopped), for all positions
at once.

`simulate_early_exit` implements this and is verified to reproduce incremental
early-exit decoding to **2.4e-07** (float32 noise), including with non-trivial
adapters.

With it, `kv_exposure="simulated"` becomes the supervised step of a DAgger-style
loop: collect states under the current policy, fit, repeat. Exit depths are
sampled uniformly rather than taken from the model's current confidence, since
the latter is a target that moves as the model trains.

## 13. Drift and refresh

`trace_decode` runs a **fixed** token sequence through early-exit decoding, so
the context is identical to a full-depth pass and any difference in the logits
comes purely from the cache. That separates cache-induced drift from the
ordinary divergence of having sampled a different token.

`refresh_every` forces a periodic full-depth token, re-anchoring the residual
stream and writing exact keys and values at that position. It needs no separate
code path: uncertainty is non-negative, so a threshold of 0 can never fire.
`refresh_every=1` exactly reproduces full-depth generation, which is the
invariant that checks the mechanism.

## References

- Elbayad, Gu, Grave, Auli. *Depth-Adaptive Transformer.* ICLR 2020.
- Schuster et al. *Confident Adaptive Language Modeling (CALM).* NeurIPS 2022.
- Elhoushi et al. *LayerSkip.* ACL 2024.
- Zhou et al. *PABEE*; Xin et al. *DeeBERT*; Liu et al. *FastBERT.*
- Raposo et al. *Mixture-of-Depths.* 2024.
