# Draw-handling candidate

This candidate corrects the repetition and tablebase defects diagnosed in the
released anneal128 blend75 engine. Its weights, book and all 96 Syzygy files are
unchanged. The opponent for promotion is that released neural engine, not the
earlier classical `05046b2` baseline.

## Behavior

- A new bounded root scan recognizes the referee's automatic claim before a
  third occurrence is actually played. It checks the candidate move and the
  opponent's next reply, including a claim by an intended following move.
- An immediate automatic draw receives the normal draw score. An opponent's
  optional drawing reply caps the move's score; it never raises a losing line
  into a guaranteed draw. Checkmate takes precedence over fifty-move claims.
- Before a cycle becomes forced, moves returning to an already visited position
  receive a maximum root score of +50 cp. Another positive continuation can then
  outrank them. Negative scores remain negative, retaining defensive draws.
  This preference is applied inside root search before updating alpha, rather
  than selecting an alternative from unverified fail-low scores afterward.
- Forced moves record their resulting positions. Obsolete history is discarded
  after irreversible moves; an observed opponent pawn move or capture also
  clears the obsolete history.
- Tablebase move selection counts an immediate pawn advance/capture as one ply
  to progress, adds one to a reversible successor's DTZ, prefers checkmate and
  incorporates the automatic-claim consequences.

The claim scan uses the existing team's bitboard move generator and state
transitions through Numba. It is warmed during import, has no disk cache, and
is charged against the move budget. A deadline retains only proven partial
claim classifications; it does not invent outcomes for unexamined moves.

## Correctness evidence

`tools/draw_regression.py` compares against the unchanged python-chess referee:
3,468 board/claim comparisons, all legal root moves in the 26 recorded draw
positions, six near-fifty-move root positions, checkmate precedence, defensive
search and tablebase draws, and history after forced moves. The recorded pawn
ending, rook mate, queen mate and bishop-knight mate all convert to checkmate.
The final candidate also passed the standard Lucena regression.

The claim scan averaged about 5 ms and peaked below 20 ms in the 26 recorded
positions. The extracted archive has 49,470,038 unzipped bytes and 107 members;
its measured init was 13.736 seconds with peak working set 228,585,472 bytes.

The selected integer model still matches both independent inference references
exactly over 10,000 held-out positions. Thirty fixed-depth comparisons without
real-game repetition context match the released search in move, score and node
count. The feature encoder, evaluator, quantized weights and blending policy
were not changed.

## Investigation and rejected attempts

Checking only imminent claims was too late in the Lucena cycle: by the last
turn, every escape let a claim occur. A version suppressing all cached search
scores after a repeated real position broke the initial loop but had unstable
conversion and less search depth. Those failed logs and diagnostic traces are
retained under the separate `draw-fix` worktree. The selected `draw-fix-fast`
candidate keeps the established search cache and adds the earlier root
preference for progress. Both its earlier standard run and the final standard
run converted Lucena by checkmate.

The final candidate scored 231/300 (77.0%) on the full WAC tactical screen at
one second per position. The previous neural release scored 228/300. This is a
tactical screen, not evidence of an Elo gain.

## Match validation

The runtime is frozen at
`37ae6ac0522dd0790fd59611750607b45423b558031e192d1fc51e658b009310`.
The released control is
`d488f913a5d059a444a599e2696db66b4037cc539fb6eb556c3e6761dd89438c`.
`data/runs/draw-validation-plan.json` predeclares 20 games at 20s+0.3s on unused
opening indices 100–109, followed by 40 games at 120s+0.5s on unused indices
110–129 if the quick screen is positive and clean. Every opening is played
with reversed colours, through suspended one-core real subprocess agents.
The quick screen finished **+5 =11 -4, 52.5%**, with all 20 PGNs replaying
legally, unchanged runtime fingerprints and zero failed outcomes or runtime
error/fallback markers from either engine. Maximum recorded init was 14.797
seconds and peak working set was 245,858,304 bytes. This is a small positive
screen, not convincing evidence of a strength gain. The predeclared 40-game
tournament-clock comparison is running.

All 11 quick draws ended by repetition. Eight last search decisions were scored
as a draw, one retained a negative score while the opponent allowed a draw, and
two came from tablebases. None of the last search decisions still claimed a
large positive score. All 25 logged claim scans completed; the maximum logged
scan was 16 ms. These are decision diagnostics. Draw rates against this neural
opponent cannot be compared directly with the older classical-opponent study.

The predecessor's separate 112-game positive SPRT against classical `05046b2`
is retained as prior evidence. It is not evidence for this patch's incremental
gain and will not be pooled with these games.
