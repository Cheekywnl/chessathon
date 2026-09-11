# Refined engine: published before full-clock validation

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

## Full-clock validation: pending

Published first at the user's request. The subsequent test uses fresh opening
pairs 30-69, 120s+0.5s, one core per game and actual opponent-time suspension.
It uses ordered complete-pair GSPRT hypotheses 0/50 local Elo, alpha/beta .05,
at least 20 pairs and at most 80 games, retaining all already running pairs.
Completed results will be appended after the run; they are not claimed here.

## Reproducibility

- GitHub branch: [hour-release](https://github.com/Cheekywnl/chess-codex-isolated-20260910/tree/hour-release)
- Source commit before publication: `35147d645d25e039aeae3a5270033930dd16ac24`
- Frozen runtime SHA256: `029bec79c7f2c75fdd0a1ca5badbee1c9aaa734c5a1675b0e6231871c46c776c`
- ZIP SHA256: `409c48e8f4aeae994be2add38f4d09a46a0e4c234d99253f01eebe18d65b4188`
- ZIP contents were checked byte-for-byte against the selected tested runtime.
- The original validated release remains available as rollback.
