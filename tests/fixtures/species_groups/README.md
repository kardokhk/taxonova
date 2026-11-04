Regression fixture for `taxonova.species_groups` (fix A, 2026-11-04).

`val_v4v5_rows.tsv`: 23 called lineages (`final`) of the frozen V4-V5 pipelines on the V4-V5 VALIDATION split
(campaign `wip/2026-11-01-dev-targets`, `integration/v4v5/work/out/<ref>/VAL_V4-V5.tsv`, job 5412094; no held-out
data), drawn with seed 20261104: per reference 4 multi-species group labels, 2 bracketed single names, 2 single
species and 1 genus-only call. `expected` and `reason` were written by an independent rule (species blank when the
species label is `[...]` containing `|`, everything else unchanged), not by the module under test.
