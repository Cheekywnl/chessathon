# Exact feature-sum cache candidate

This candidate follows the frozen compiled engine that scored +17 =2 -1 in 20
short games against released 7eda0fa. It has not yet demonstrated an additional
playing-strength gain. The next 20-game paired screen uses unused indices 182-191
at 20s+0.3s, with all results retained. Its opponent remains released 7eda0fa.

## Implementation

One private cache per Search remembers the last evaluated board and both raw
HalfKP feature sums. Exact piece-set differences update the sums across children,
siblings, returns to parents, and null moves. A king-anchor change refreshes that
perspective. Raw sums use int32; activations are clipped only after all changes.
No cache state is shared with the background worker or the transposition table.

Profiling found full transformer refresh expensive on unrelated held-out boards.
Actual search measurements were less dramatic: an always-on cache helped crowded
positions but slowed sparse ones, so the final path uses it only with at least 24
pieces. An attempted dense-helper extraction also slowed sparse positions. The
final code preserves the original full-refresh function verbatim, with a separate
dense helper for cached sums. Both are checked against the integer reference.
The earlier ungated and extracted-helper logs are preserved as rejected attempts.

## Verification

- 50,111 legal transitions and 105,643 parent/child/return/null position checks per
  width 128 and 256: every raw sum and output matches full refresh. Both colours,
  castling, en passant, promotions, capture promotions and king refreshes covered.
- 10,000 independent integer-reference and bitboard inference checks pass.
- Final code matches the preceding compiled runtime in all 56 fixed-depth 6 search
  cases, every table slot, move/score/rank/node count, killer/history/repetition
  entry, 32 forced abort states, and 1,944 mate/key/table boundary probes.
- The older gated version also passed 84 persistent-table/aspiration comparisons.
  The final code restores the old full-refresh function and leaves cache logic,
  persistent state and root search unchanged.
- Final make gate and Lucena pass. Full one-second WAC is 274/300 (parent 273/300);
  the earlier extracted-helper version scored 263/300 and was superseded.
- Final alternating-order timings: 6.169s before versus
  5.673s after across all blends (1.087x).
  The actual blend 75 subset is only 1.056x. This is a modest speed improvement,
  not a measured Elo gain. Some individual cases still ran slower.
- Extracted package: 44,998,818 bytes, 109 files,
  43.987s local one-core initialization,
  474,075,136-byte peak working set, legal smoke moves.
- ZIP SHA256: e63d12e547a7f76f14a1e0ddb1928387173f84edf3346260f129963c862c4f3c
- Runtime SHA256: 79df5b2416394fbfc947307ee6208497f6c74429baf65942c76ca54dd2172e51

The selected released width 128 weight, all opening entries and 96 tablebase files
are unchanged. Width 256 was used only as an extra correctness check; its unselected
weight and raw training checkpoints stay outside Git. Main and the existing
recommended upload are unchanged pending the agreed release process.
