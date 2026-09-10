# Forced-reply draw fix and targeted training, 11 September 2026

Status: correctness screens passed for the updated draw-fused candidate. The short real-clock
comparison is in progress. This remains an experimental branch, not a promoted release.
The target is 90% actual wins against released neural 7eda0fa; it has not been demonstrated.

## Concrete draw failure and fix

The rook-and-pawn trace stayed objectively winning according to offline Syzygy, but failed
to reduce the distance to conversion. Eventually Kc8 allowed ...Rc2+, forcing Kb8, after
which the opponent could claim repetition by the intended ...Ra2. The old root scan only
saw three hypothetical plies. The forced fourth ply was outside its horizon.

The compiled scan now extends exactly one forced reply and classifies that opponent option
as a draw ceiling. It does not invent a guaranteed draw for a losing side. The fifty-move
screen starts at halfmove 96 to account for the extra ply. Root timing remains bounded.
The 7.2% measured search refactor from the previous checkpoint is retained unchanged.

Validation on the current source and released weights:

- 3,468 referee comparisons, 26 previously recorded draw roots, nine fifty-move roots,
  and four colour/file variants of the new forced-reply trap pass.
- Defensive draw retained; optional opponent draw still leaves a losing score (-1159).
- Existing pawn, rook, queen and bishop/knight tablebase conversions pass.
- Largest recorded root scan 18.155 ms; mean 5.313 ms over the recorded draw cases.
- Standard Lucena regression now reaches checkmate. At the previous trap it selects Rd4,
  constructs the bridge, promotes, and finishes through existing tablebase coverage.
- make gate: ruff and mypy (48 files), two legal smoke-game checkmates, pass.
- Full WAC: 231/300 at one second/position. These are wall-clock screens, not strength proof.
- Previously completed 52,009 transition and 30 fixed-depth search parity checks apply to
  the unchanged fused state/search code. The released network and inference are unchanged.
- Extracted package audit passes: 49,472,715 unzipped bytes, 107 files, 11.228 s init,
  peak working set 230,256,640 bytes, three legal smoke moves, permitted source/assets only.
- Package SHA256: cfaa3c58ba191a4585372f10d0c3c73b19bcdcafddb8b5c722e60cc8e29c57b2
- Runtime SHA256: c91d08b697d2b469a2fe766b09d698e43c20438ce714fa525c5647d013486e64

The fixed short screen uses fresh indices 142-151 from the recorded opening pool, both
colours, 20s+0.3, one physical CPU per worker and real opponent-time suspension. All 20 games
will be retained. Only >=70% score with >=60% actual wins warrants considering a longer
test. That screen threshold does not satisfy the separate 90% actual-win objective.

## Targeted training

No broad dataset was downloaded. The existing 148,547,121-row training split and 1,498,177-row
held-out split remain disjoint. Every complete epoch visits all training shards. The new
objective weights 8-12-piece positions by 4, 13-20 by 2, and other positions by 1. It uses the
same architecture and continues only the team's recorded random-initialization lineage.
Selection minimizes weighted held-out MSE, with an overall-MSE regression ceiling of 2%.

The initial 100k pilot hit a reporting-key collision after training; it was fixed and the
fresh pilot completed with bounded RAM. Phase boundaries and 1,000 encoded piece counts
were independently checked against python-chess. No held-out records enter training.

The 0.0001 learning-rate run stopped after 50,023,145 row presentations without improving
the objective, retaining the parent. A 0.00002 run stopped after 160,079,574 presentations
and selected its 110,051,881-row checkpoint:

- Overall held-out MSE: 0.0106574617 -> 0.0106540493 (about 0.032% lower).
- 8-12-piece held-out MSE: 0.0200707505 -> 0.0199841031 (about 0.43% lower).
- Weighted held-out MSE: 0.0130994693 -> 0.0130760516 (about 0.18% lower).
- The improvement is small. No claim of Elo or a large strength gain follows from it.
- Training time 244.068 s; >8 GB RAM remained available in the final resource snapshot.
- Checkpoint SHA256: 360bb8bcb9f322e58dadc32a4272a352eaedeb518e3c614232e7ede8c0de40c5
- Float SHA256: 4a43820781eb8c8fca122d87144ca750d47dae22bc46c608e80fc5057b3964f3
- Quantized SHA256: 598b8ec012b0339c73848c471e0279064d2aea784dbc34868e522b67a2df2779
- Quantized size 6,942,202 bytes; int16 bound 15,191.
- 10,000 held-out integer-reference and bitboard-inference checks: zero mismatches.
  Mean quantization error 5.064 cp, p99 14.793 cp, max 32.998 cp.

This checkpoint is outside the active match candidate. Neither it nor the raw checkpoints
are committed as a release. Its provenance and training code are preserved for review.

## Offline tablebase diagnosis

KRPvKR WDL/DTZ requires 30,104,032 bytes, before missing promotion dependencies. The six
additional files total 41,840,864 bytes; they do not fit alongside the submission's current
assets. They were downloaded only into work/endgame-oracle, verified against the Lichess
mirror's SHA256 list, and used after engine decisions for diagnostic labels. The playing
agent does not access that directory. Existing shipped tablebases and book are unchanged.
Sources: https://tablebase.lichess.ovh/tables/standard/ and the canonical competition rules
https://aichessathon.com/docs/rules.md (freshly fetched for this refinement).
