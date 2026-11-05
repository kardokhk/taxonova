Regression fixture for `taxonova.species_groups` (written 2026-11-04).

`val_v4v5_rows.tsv`: 7 called lineages (`final`) of the frozen Greengenes2 V4-V5 pipeline on the V4-V5 VALIDATION
split (no held-out data), drawn with seed 20261104: 4 multi-species group labels, 2 single species and 1 genus-only
call.
`expected` and `reason` were written by an independent rule (species blank when the species label is `[...]`
containing `|`, everything else unchanged), not by the module under test.
