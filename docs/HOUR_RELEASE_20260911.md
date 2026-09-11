# Penultimate engine: full-clock validation complete

Use **submission-refined.zip** for this candidate.

- Short screen against released 7eda0fa: **19 wins, 1 draws,
  0 losses in 20 games**: 95.0%
  actual wins and 97.5% score.
- The combined refinement's direct screen against the strong cache parent:
  **9 wins, 4 draws, 7 losses**,
  55.0% score. The predeclared rule selects **refinement-hour**.
- All short-test PGNs replay legally. Runtime fingerprints stayed fixed; neither
  side produced failures, fallback diagnostics, illegal moves or time losses.
- Gate, endgame, neural equivalence, draw and package checks passed. The combined
  candidate scored 270/300 WAC; its cache parent scored 274/300. Detailed results,
  limitations and rejected trials are retained in the evidence archive.

## What changed

This release includes the team's compiled recursive search and corrected draw
handling, plus the exact neural feature cache. The combined candidate also removes
inactive repetition entries and disables pondering. The trained width-128 model,
opening choices and Syzygy assets are unchanged. No published network is used.

The measured short-game result meets the requested 90% actual-win fraction in
this sample. It does not guarantee a future win rate or establish a platform Elo.

## Full-clock validation: complete

Against the older **already-NNUE** release `7eda0fa`, this exact ZIP finished
**42 wins, 15 draws, 1 loss in 58 games**:
85.3% score and 72.4% actual wins at 120s+0.5s.
This is a clear improvement over that older release. The requested 90% actual-win
fraction was **not** reached in this longer-clock sample.

The predeclared paired GSPRT (0/50 local Elo, alpha/beta .05) accepted H1 after
23 complete pairs, LLR 3.072927. All already
running pairs were retained: 29 pairs total, final LLR
3.874424. The descriptive logistic score estimate is
about +306 local comparison Elo; it is not a platform rating or a confidence
bound, and the earlier short-game score overstated the longer-clock margin.

All PGNs replay legally. Both runtimes stayed byte-for-byte fixed. There were no
flags, illegal moves, crashes, fallback messages or other diagnostics. Maximum
recorded peak memory was 481,685,504 bytes and maximum
initialization was 64.015s. The only loss was
a normal checkmate. Of the 15 draws, 14 were repetitions and one the fifty-move
rule; 14 final search decisions scored the draw at -20cp and one used a tablebase.
None ended with a final search move still scored at least +150cp, and every logged
draw-claim scan completed.

The user's subsequent five-hour refinement targets further improvement against
**this penultimate runtime**, not against `7eda0fa`. New experiments are separate.

## Reproducibility

- GitHub branch: [hour-release](https://github.com/Cheekywnl/chess-codex-isolated-20260910/tree/hour-release)
- Source commit before publication: `35147d645d25e039aeae3a5270033930dd16ac24`
- Frozen runtime SHA256: `029bec79c7f2c75fdd0a1ca5badbee1c9aaa734c5a1675b0e6231871c46c776c`
- ZIP SHA256: `409c48e8f4aeae994be2add38f4d09a46a0e4c234d99253f01eebe18d65b4188`
- ZIP contents were checked byte-for-byte against the selected tested runtime.
- The original validated release remains available as rollback.
