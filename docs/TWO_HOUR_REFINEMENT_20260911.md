# Additional refinement window, 11 September 2026

The starting engine is the corrected c1b6dc4 submission, runtime fingerprint
`99807db7aef6e128c8b1480b5baa5e46a7df821951ea2c87f47785aaec9a8900`.
Its ZIP SHA-256 is `4d3086d962a7a1760a0e7388fe31f267937c893542693d6a26eb4d67c4712110`.
The budget is 50,000,000 fully expanded bytes; the starting artifact uses 48,999,996.
The user authorized additional work from 07:09 to 09:09 UTC, targeting another
200 Elo or more. Targets are not measured results.

First selection experiments compare network blend 90%, guarded adaptive null
search, and bounded history with promotion ordering against that exact starting
engine. Each uses 40 games at 8s+0.12s, with colour-swapped opening indices
180-199 from the existing reproducible pool. The same positions across experiments
make comparisons easier; results from different experiments must not be pooled.
Private selection workers reuse compilation and reset all game state. Final
confirmation will use fresh processes and separate openings.

Two exact arithmetic-loop alternatives were discarded: their hidden activations
and outputs matched all 4,000 cases, but median dense-loop times were 0.840s and
1.236s versus the current 0.820s. This is a timing result, not an Elo measurement.
Further work includes broader exact KRP-versus-KR policy coverage, with independent
generated seeds, WDL/DTZ checks, preservation of all existing policy entries, and
the same root repetition/clock guards. No third-party engine or network ships.

The earlier release's completed tests are archived separately in
`validation/refinement-final-20260911`: larger-book full-clock 29W/20D/7L,
corrected exact-ZIP quick 10W/6D/4L. Neither is a gain measured during this new window.
