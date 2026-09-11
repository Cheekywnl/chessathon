# One-hour refinement candidate

The frozen candidate combines the exact feature-sum cache with two additional
changes: inactive repetition entries are removed on recursion return, and the
legacy ponder worker is no longer started. The search's evaluation, pruning,
ordering, model, opening entries and Syzygy assets are otherwise unchanged.

Repetition detection needs positive counts from real history and the active
search path. Retaining a zero count for every visited leaf grew the dictionary
to 409,059-868,871 entries in three depth-11 probes. Removing them leaves 24-29
entries (including harmless zero entries for root moves), while preserving every
move, score and node count. The separate deep runs took
27.337s before and
24.145s after (1.132x); process RSS at the end
was 618,524,672 versus 518,963,200 bytes. These three runs were sequential rather
than alternated, so the timings are exploratory. In the alternating 56-case
depth-6 comparison, the changed version was slightly slower: 6.356s versus 6.226s.
The useful memory reduction is clearer than a universal speed claim.

## Completed checks

- 56 complete searches match the separately loaded prior compiled backend and
  inference source: moves, scores, ranked root moves, node counts, full table,
  killers/history and every positive repetition count. Zero entries are omitted
  from comparison because deleting them is the intended change.
- 32 forced interruptions preserve exact active repetition/table/heuristic state;
  1,944 table/key/mate boundary comparisons pass; deadline and cancellation pass.
- 84 iterative-deepening/aspiration/persistent-table comparisons against the
  independent Python search pass on the final combined runtime.
- Final make gate passes (56 typed files and two legal checkmate smoke games).
  The first attempt only failed a long line in the validation tool; its log is
  retained and the formatting fix changes no engine behavior.
- Lucena converts to checkmate. The draw suite passes its 3,468 referee checks,
  26 recorded roots, nine fifty-move roots, four forced-reply mirrors and four
  tablebase conversions. The inherited terminal fixes are unchanged.
- WAC is **270/300 at one second**, versus the cache parent's 274/300. This is a
  small tactical regression in a time-limited suite and is explicitly retained;
  direct match evidence determines whether the combination advances.
- Actual ZIP audit passes: 44,998,813 bytes, 109 files, 42.194s one-core import,
  peak 476,282,880 bytes and legal smoke moves.

Runtime SHA256: 029bec79c7f2c75fdd0a1ca5badbee1c9aaa734c5a1675b0e6231871c46c776c

ZIP SHA256: 409c48e8f4aeae994be2add38f4d09a46a0e4c234d99253f01eebe18d65b4188

## Match plans and prior evidence

The exact cache parent completed +18 =2 -0 against released 7eda0fa in 20 games:
90% actual wins, 95% score. The independent no-ponder trial completed +8 =8 -4
against the stronger compiled parent: 60% score. These different trials are not
aggregated or treated as a guarantee for the combined candidate.

The combined build is frozen for two predeclared 20-game screens at 20s+0.3s:
fresh opening indices 10-19 against released 7eda0fa, and 20-29 directly against
the 18-win cache parent. All games and colour pairs will be retained. Full-clock
tests remain pending until after the user's requested hour-end publication.

## Rejected experiments

The combined null-cutoff gate and unreduced checking-move trial passed compiled
versus independent Python parity and draw/endgame checks, but WAC fell to 260/300.
Separated trials scored 268/300 for the null gate and 263/300 for check depth.
All are rejected, with patches/logs preserved and none of their changes shipped.
Doubling table capacity from 2**21 to 2**22 took 19.047s versus 19.020s across 12
depth-9 searches, with no useful improvement; capacity remains unchanged.
The already archived packed-table and width-256 trials were also rejected.

A scan of 171 already played games identified missing small tablebase classes.
Difficult covered-material gaps require more than the available space; smaller
additions mostly cover existing easy conversions. No new tables were added.
No new training data was downloaded and no unselected weight was promoted.


## Completed short screens and release choice

Against released 7eda0fa: **19 wins, 1 draws,
0 losses; 95.0% actual wins,
97.5% score**, in all 20 games and ten colour pairs.
Peak process memory was 479,227,904 bytes and
maximum import 51.375s. Every PGN is legal,
both sides have zero runtime diagnostics, and fingerprints stayed fixed.
The one draw was recognized at -20cp with completed claim scans, rather than
being evaluated as an unresolved win.

Directly against the cache parent: **9 wins,
4 draws, 7 losses; 55.0% score**.
All 20 games and ten pairs are retained, with legal PGNs and zero diagnostics.
This comparison uses fresh openings 20-29 and is not mixed with the weaker
released-opponent results.

The predeclared selection rule chooses **refinement-hour**. The actual model remains
the previously validated team-trained width-128 model. An hour-release branch
and exact ZIP will be published before the long test as the user requested.
The full-clock test uses untouched openings 30-69, 120s+0.5s, ordered complete
pairs, GSPRT hypotheses 0/50 local Elo, alpha/beta .05, at least 20 pairs and at
most 80 games. Already running pairs are retained after a boundary. This tests
playing strength and operational reliability; it establishes no platform rating.
