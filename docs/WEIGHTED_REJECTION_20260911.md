# Reject the targeted phase-weighted model

This trial combined the fused search and forced-reply draw fix with the team's new
phase-weighted width-128 checkpoint. Only the model differed from draw-fused.
Its provenance and full training logs are in the inherited forced-reply evidence.

The candidate passed 10,000 integer-reference comparisons, integration, gate, Lucena,
234/300 WAC and an extracted-package audit (49,472,153 unzipped bytes). Its runtime was
c04155d73816ce8cf497b95d0c31ca390a6301eb75ca0d68b71d00a06c6c50f1. The tested quantized
weight SHA256 was 598b8ec012b0339c73848c471e0279064d2aea784dbc34868e522b67a2df2779.
Package SHA256 was b461e96a43f7f11b413568ce0ab2ca71333738792ef7e39a0b18f5a552e8f6cc.

It used previously unused opening indices 152-161, with both colours. The improved
fitting metric did not improve this match screen. The unselected weight and raw
checkpoints remain outside Git, as required. This branch records the failed experiment;
its committed inherited weight is the previous released model, not this rejected trial.

## Completed short comparison

All 20 games and 10 colour pairs completed at 20s+0.3s against frozen released neural
7eda0fa. Result: **+6 =8 -6,
50.0% score, 30.0% actual wins**.
Every PGN replays legally, all runtime fingerprints remain fixed, and neither engine
logged runtime failure/fallback diagnostics. Peak working set was
245,739,520 bytes; maximum import time was
15.046 seconds.

This fails the declared promising gate and is far short of the 90% actual-win objective.
No long tournament is launched. Main and the recommended upload remain the validated
7eda0fa release. A selected short opening sample is not a platform rating measurement.
