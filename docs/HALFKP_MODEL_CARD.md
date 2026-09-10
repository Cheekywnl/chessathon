# HalfKP model and submission provenance

This document describes the selected `full-128-anneal` network. It is released on
the practical evidence of 60 completed independent 120s+0.5s games: 35 wins,
21 draws and four losses. The full SPRT continues as a supplementary check and
has not yet passed. This is not a claim of a platform rating. The release report
records the decision basis and submission hash.

## Training origin

The team trained this model from random initialization on this machine on
10 September 2026. The initial `full-128` run used seed 20260910; the annealing run
continued only that team's own checkpoint. No published network, third-party
engine implementation, or third-party weight file was used in the network.

The existing Lichess evaluation CSV supplied 150,000,000 rows; the ten existing
team self-play CSVs supplied 56,927 rows. The bounded loader rejected 11,629 rows
with more than 30 non-king pieces. It accepted 150,045,298 rows, with 148,547,121
training rows and 1,498,177 held-out rows. These are row counts, not a claim that
every position is unique. The dataset was already present and was not downloaded
again. The source is the [Lichess evaluation database](https://database.lichess.org/#evals),
distributed under CC0, and the team's existing self-play data.

CSV centipawn labels are White-relative. Training converts them to side-to-move
probability targets with `sigmoid(cp / 400)`. Self-play labels are White's WDL
result, complemented when Black is to move. The stable split hashes the first
four FEN fields with BLAKE2b, digest size 8, personalization `halfkp-split-v1`,
little-endian modulo 1000; buckets 0 through 9 are held out. Identical canonical
positions cannot cross the split, even between source files.

Preparation used 250,000-row shards, with one current shard and current batch in
memory. Each complete epoch shuffled all training shards and their rows.
The initial run presented 180,081,830 training rows in 375.845 seconds; the selected
continuation presented another 297,094,242 rows in 600.240 seconds. Repeated passes
are included in those presentation counts. Full held-out probability MSE was
0.010657461710659292 for the selected checkpoint. Loss selected checkpoints;
actual games selected the deployment candidate.

Training used an RTX 3070, CUDA PyTorch 2.5.1+cu121, batch size 8192, Adam and a
cosine learning-rate schedule. The continuation started at learning rate 0.0002
and ran two complete epochs. Exact commands, source hashes, run metadata and
quantization results are preserved in `halfkp_training_provenance.json` alongside
this document. Raw data, optimizer checkpoints and float exports remain local
and are excluded from Git and the submission.

## Architecture and inference

The shared transformer has 40,960 king-relative HalfKP features and width 128.
Black uses the vertically mirrored perspective, and the side-to-move accumulator
comes first. The clipped dual-perspective output feeds a 32 -> 32 -> 1 dense head.
A randomly initialized 640-feature piece-square factor was used only in training
and folded into the king buckets before quantization.

The shipped transformer and accumulators use int16; dense weights use int8, with
integer sums and clipped activations. Both perspectives are recomputed at each
evaluation. The final scalar converts to centipawns. There is no CUDA or PyTorch
inference path in the submission. The conservative int16 accumulator bound is
15,191, below 32,767.

On 10,000 held-out positions, the independent integer reference and bitboard path
each had zero mismatches. Float-versus-quantized absolute error averaged 5.250 cp,
with p99 19.344 cp and maximum 41.234 cp. Probability MSE on that subset was
0.0094173349 for float and 0.0094192923 for the integer model.

The selected engine blends 75% neural evaluation with 25% classical evaluation.
It retains the classical route for seven or fewer pieces and bare-king endings,
and retains the established mop-up helper, repetition safeguards and all original
Syzygy files. A 12,500,000-byte book makes room for the network; book lookup ends
after move 20. The evaluated candidate includes this book tradeoff.

## Exact assets

| Item | Bytes | SHA-256 |
| --- | ---: | --- |
| `weights/halfkp.npz` | 6,942,764 | `3ab6d135ecdd3b9b24b6c37aace3e9a658c3cc8c6c2093dee1a02edb1a0401a2` |
| `book/codekiddy.bin` | 12,500,000 | `16a2ab58fb6793e6749a23e1a59ee2e4fc2af5800849c3ba9dafaa94b190f88b` |

Selected checkpoint SHA-256:
`85fe3648e14711f989b85bc590499cbe6b12e5a8aa39262ac0c81cf877234e0a`.
Selected float export SHA-256:
`045f8db4f39e92dc1f146d3caf5f42c96d7845a2fdcafb9320e5d420ecd1565e`.
Runtime source-and-assets fingerprint:
`d488f913a5d059a444a599e2696db66b4037cc539fb6eb556c3e6761dd89438c`.

## Scope of validation

The 20-game selection screen at 20s+0.3s scored 14 wins, 5 draws and 1 loss against
baseline `05046b2`, using paired colours over ten openings. A separate four-game
check at 120s+0.5s scored two wins and two draws. Neither sample is included in
the independent final SPRT.

Final testing uses real subprocess agents, one physical CPU core per simultaneous
game, the competition-pinned CPU interpreter, and actual process-tree suspension
between turns. Fresh paired openings and the sequential-test bounds were frozen
before any final game. Logs retain full PGNs, resource measurements, runtime
diagnostics and source/asset fingerprints.

Local Windows testing does not establish a platform Elo or substitute for the
platform's Linux upload validation. The final release report identifies the
completed game evidence and any remaining limitations. The clean original
`05046b2` opponent and the user's existing checkout remain preserved.

The subsequent 76-game draw investigation found that the existing repetition
safeguard can act too late for automatic draw claims and that tablebase move
selection can shuffle instead of making a winning pawn advance. These conversion
defects remain in the released runtime. See `docs/DRAW_ANALYSIS.md` for the
reproductions and phase-specific held-out error measurements. Training data alone
cannot correct a repetition-history or tablebase-selection defect.

The completed independent test accepted H1 after 53 paired openings. Including
the remaining in-flight games, this unchanged build scored +62 =43 -7 in 112
games at 120s+0.5s against `05046b2`, with zero failed outcomes or runtime error
markers. This supports a local improvement; no platform rating is established.
