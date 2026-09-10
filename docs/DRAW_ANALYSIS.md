# Elo, drawn games and targeted training

The reviewed snapshot scored **45 wins, 26 draws and 5 losses in 76 games**
against baseline `05046b2`, at 120 seconds plus 0.5 seconds per move. That is
76.32% of the available points: approximately **+203 local logistic Elo**.
The draw rate is 34.21%; the candidate won 45 of the 50 decisive games.

This is a descriptive estimate for these opponents, openings, clocks and this
machine. It is not a platform rating or a prediction against every opponent.
A 50,000-resample bootstrap over the 38 opening pairs gives an approximate
145–266 Elo range. That range is not adjusted for sequential stopping or
repeatedly inspecting the test. The formal SPRT is a separate decision about
its predeclared hypotheses; its +20 hypothesis is not an estimate of the gain.

The underlying 76-game snapshot SHA-256 is
`17956b1c519a19da1167e2fb1f01ccb4c8b5df8ac72694a95ce06baa5768daec`.
Later games can change the estimate. The released runtime remains
`d488f913a5d059a444a599e2696db66b4037cc539fb6eb556c3e6761dd89438c`.

## What happened in the draws

All 26 ended through the unchanged referee's `threefold_repetition` rule.
None ended through a flag or the 600-ply limit. The last recorded candidate
search clock was above five seconds in every game; this does not by itself
exclude an effect from limited search time.

Of the 25 draws whose last candidate move came from search, **18 still had a
reported score of at least +150 cp, including 15 at +300 cp or more**. The other
seven contained six scores between -150 and +150 cp and one score of -592 cp.
The remaining draw came from the tablebase route. Thus, avoiding every draw
indiscriminately could throw away useful defensive draws.

Search scores are the engine's estimates, not proof of a forced win. The log
also reports the search score before any final repetition override. The initial
count of 19 games above +150 included a stale +1300 search score from the game
that subsequently entered tablebase play; its final move had no search score.
The more precise split above separates that game.

### The repetition safeguard acts too late for this referee

The referee calls `board.outcome(claim_draw=True)` before asking for the next
move. This can finish the game when the player to move has a legal move that
would create a third occurrence, even though that move has not been played.

All 26 final boards had a claim available but had not yet reached an actual
third occurrence. In nine games this happened immediately after the candidate's
move. In seventeen it happened after the opponent's reply, before the candidate
got another turn.

The candidate's final safeguard asks whether its chosen move itself reaches an
already twice-recorded position. Reconstructing the actual candidate history and
calling that predicate returned **false on all 26 last candidate moves**.
This confirms a mismatch in timing. It does not prove that all 26 positions
offered a sound way to avoid the draw. The referee was not changed.

For example, in opening pair 1 as White, the candidate shuffled its bishop
between a3 and b2 while Black shuffled its king between h6 and h7. It was
approximately five pawns ahead in material; the last two engines' evaluations
were +986 and +845 cp from the candidate's perspective. After 46...Kh7, the
referee ended the game because 47.Bb2 could repeat. Waiting to reject 47.Bb2
could never work: the candidate was not given that turn.

### A concrete missed win in tablebase play

In pair 6 as Black, the ending was Black king and a-pawn versus White king.
Before 77...Ke8, the local Syzygy tables report a win (WDL +2) and distance to
the next pawn move or capture of one ply (DTZ 1). Advancing **77...a4** preserves
that win and breaks the repetition.

The existing selector instead chose **77...Ke8**, reproducing the recorded
game exactly. It compares the successor positions' absolute DTZ values without
crediting a pawn move or capture that resets the counter immediately. Ke8,
Kf7, Ke7 and a4 all received the same progress rank of -2, and move order chose
the king shuffle. The referee then declared a repetition draw.

This is a verified conversion defect in the tablebase move selector. Training
the neural network on more pawn endings would not fix that selector; the
network is bypassed in this position.

## What more training data might help

A deterministic sample of **601,000 training rows across all 601 shards** and
**120,200 held-out rows across all 601 validation shards** gave:

| Pieces on board | Share of sampled training rows | Neural prediction MSE on held-out labels |
| --- | ---: | ---: |
| 21–32 | 59.52% | 0.00681 |
| 13–20 | 25.88% | 0.01438 |
| 8–12 | 10.13% | 0.02012 |
| 2–7 | 4.47% | 0.02184; neural route bypassed here |

These are errors against the existing sigmoid-transformed evaluation labels,
not errors in game outcomes or proof of insufficient data. Material bands have
different difficulty and label distributions. Still, the nearly threefold gap
between 8–12 and 21–32 pieces identifies a useful area for a training experiment.
The existing corpus already contains many such positions, so a larger download
is not automatically the best first step.

Recommended order:

1. Test corrections to repetition-claim timing and tablebase progress selection
   as a separate candidate. Preserve defensive draws and the released build.
2. Try giving existing **8–12-piece positions and positions with substantial
   advantages** more training weight while retaining a broad mixture. Study
   conversion, passed pawns, simplification, king safety and perpetual-check
   escapes. These are candidate themes, not confirmed missing-data categories.
3. If those tests still show a coverage gap, obtain new diverse positions with
   reliable deep evaluations or exact tablebase outcomes where applicable.
   Queenless rook positions alone did not have unusually high aggregate error
   in this audit, so the evidence does not support making them the sole focus.
4. Keep separate held-out positions and new opening pairs for any changed model.
   Do not label every position from a drawn game as intrinsically equal: the
   tablebase example shows why that would teach the wrong lesson.

Combining evaluation labels with game outcomes is an established option in
[NNUE training documentation](https://official-stockfish.github.io/docs/nnue-pytorch-wiki/docs/nnue.html#using-results-along-the-evaluation).
For additional evaluation data, [Lichess documents varying depths and recommends
the deepest evaluation's first principal variation](https://database.lichess.org/#evals).
Neither option supplies a measurable Elo gain until a newly trained candidate
wins fresh real-protocol tests. No new training or runtime change was made for
this investigation.

The diagnostic archive contains the reviewed PGNs, captured logs, move-by-move
analysis, tablebase probe results, phase/error profile and analysis scripts.
