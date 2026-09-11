# Selected HalfKP model: moderate broad-teacher continuation

This release ships the team's own width-128 network after an additional offline
training pass. Its random-initialization lineage is preserved; no published chess
network was used. The previous model's complete training account is preserved in
[the parent model card](HALFKP_PARENT_MODEL_CARD_20260910.md).

## Data and training

The parent used 148,547,121 training rows and 1,498,177 held-out rows from the
existing Lichess evaluation data and team self-play. These are rows, not a claim
of unique positions. Parent checkpoint SHA-256:
`85fe3648e14711f989b85bc590499cbe6b12e5a8aa39262ac0c81cf877234e0a`.

The new collection contains 51,173 deduplicated positions: 13,933 from the team's
previous games and 37,240 sampled from Lichess positions, labelled offline by
Stockfish 19 at 12,000 nodes per position. Selection requires a quiet principal
move, no check, depth at least six and absolute evaluation at most 1,800 cp.
Invalid positions and impossible promotion budgets are rejected. The engine
binary stays outside the submission and private Git repository; only the
team-trained quantized model ships.

The original canonical-FEN hash split is unchanged: 50,663 new training positions
and 510 held-out positions. Original-data replay draws a 999,968-row proportional
sample across the original shards. Every epoch mixes one teacher example with
three original examples. All 1,498,177 original held-out rows are evaluated when
selecting the continuation, alongside the 510 new held-out positions.

On RTX 3070 with CUDA PyTorch 2.5.1, this continuation ran 60 epochs and
12,159,120 example presentations in 138.968 seconds. Adam used learning rate
0.00003, with transformer learning rate scaled by 0.25 and a reset optimizer.
The selected epoch minimizes new held-out MSE subject to an original held-out
regression of at most 2%. Selection chose epoch 60:

| Probability MSE | Parent | Selected |
| --- | ---: | ---: |
| New held-out positions | 0.0113766109 | 0.0097246600 |
| Full original held-out set | 0.0106574617 | 0.0107873768 |

A more aggressive width-128 continuation and a width-192 continuation were also
trained. Both lost their paired selection matches against this model and were
rejected. Lower training loss alone did not select the final engine.

## Architecture and runtime

The shared HalfKP transformer has 40,960 king-relative features, width 128 and a
32 -> 32 -> 1 dense head over the two perspectives. Training-only piece-square
factorization is folded into the transformer. Features/accumulators use int16;
dense weights use int8 and integer sums. Full-range quantization has a conservative
absolute accumulator bound of 15,221.

Inference uses Numba on one CPU core. Search reuses incremental king-bucket
accumulators and exact static-evaluation cache entries. The blend remains 75%
neural and 25% classical, with the classical route for seven or fewer pieces and
bare-king endings. There is no GPU, Torch model inference, downloaded data or
third-party engine executable in the submission.

Independent reference and bitboard inference agree on 10,000 held-out positions.
Quantization error averages 3.612 cp, p99 13.500 cp and maximum 26.407 cp.
This is numeric validation, not an Elo estimate.

## Exact lineage and reproduction

| Artifact | SHA-256 |
| --- | --- |
| Original dataset manifest | `6d63c217336798ec31b4b881f938d4d274eecc587678fad202b3b48e58df22b3` |
| New teacher manifest | `e043474e44b3b3b35b3fc7d96372ad6fc83e64a729b7dd223b10ed5c2551dfc9` |
| Selected checkpoint | `c7ebdc7fa9211e06e8d05223b3379a02ef59665564f66258a5f8d86dcf31f74b` |
| Selected float export | `75f28428894f0f75456ff20ec20fcf61811b4fe10b2f2d71b09bbb47d6c4f40d` |
| Shipped weights/halfkp.npz (6,942,559 bytes) | `c3207d459bb089de8ba4207c4189d5e04da759f97941d4bd2f6b62d3f3bed768` |

[The evidence archive](validation/refinement-final-20260911/SHA256.json) binds raw training metrics,
quantization metadata, annotation provenance, source scripts and checks to their
original bytes. Reproduction sources under `reproduction/tools/*.py.txt` retain
the exact tested scripts; restore their `.py` names into the tools package to run
them. Raw data, annotation executable, optimizer checkpoints and float weights
remain local, outside Git and the submission.

See [the release validation](FINAL_REFINEMENT_20260911.md) for games. Old results
against classical or earlier neural versions do not establish a new gain over
the penultimate release.
