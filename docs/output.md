# Output and thresholds

## The table

`taxonova classify` writes a tab-separated table with one row per input
sequence, in input order:

| column | content |
| --- | --- |
| `sequence_id` | the FASTA identifier (the header up to the first space) |
| `domain`, `domain_confidence` | label and confidence at domain |
| `phylum`, `phylum_confidence` ... `genus`, `genus_confidence` | the same for each rank |
| `species`, `species_confidence` | label and confidence at species |

Confidences are printed with four decimals. Names follow GTDB, the taxonomy of
all current models (see [models.md](models.md)). GTDB uses suffixes such as
`_A` for genera split from a historical genus and placeholder species names
such as `Prevotella sp002251435`. Some Greengenes2 names also end in a number,
for example `Pseudomonadales_A_641035`. The number is part of the Greengenes2
name and separates groups in the Greengenes2 tree that share a GTDB name.

The `taxonova classify` and `taxonova classification` commands and the Python
function `taxonova.classify` give the same table for the same model and
thresholds. On a different processor, a confidence can differ in the fourth
decimal because of floating-point rounding; labels are the same. The web app
gives the same rows too, except that it first screens
each sequence for rRNA and reports a sequence that fails the screen as
`Unclassified` at every rank.

## Unclassified

At each rank, a label is reported only if its confidence reaches the threshold
for that rank. Otherwise the rank reads `Unclassified`, and so does every rank
below it. A sequence assigned to a genus but not to a species therefore has a
genus name and `Unclassified` at species. This is the model abstaining rather
than guessing, and it is expected for short regions in which related species
have identical sequences.

The confidence is printed even for `Unclassified` ranks, so you can see how
close a call was.

`Unknown` means the reference itself had no name at that rank for the
predicted lineage.

## Short reads

Taxonova classifies reads with at least 80 valid bases (A, C, G, T or U). A
read with fewer, or with more than 5% ambiguous or other characters (N, IUPAC
codes, gaps), is `Unclassified` at every rank. Reads with 80 to 119 valid bases
are classified with the usual thresholds, and the run log (and the web app)
says how many there are.

Expect fewer and less reliable calls at genus and species on short reads. On
real V4 mock-community reads of about 253 bp cut to their first 90 bases, the
share of correct calls (precision) was 0.96 at phylum, 0.83 at family, 0.81 at
genus and 0.61 at species, against 1.00, 1.00 and 0.96 at family, genus and
species at full length. Fewer reads were named: 70% at genus and 4% at species
(98% and 79% at full length). At 120 bases, genus and species precision were
0.95 and 0.96. On V4 reads that were about 101 bp as sequenced, genus precision
was 0.88 and species precision 0.74. The reads came from organisms with close
relatives in Greengenes2; reads from less well known lineages may do worse. The
operating points were fitted on full-length amplicons. For the other regions, the first 90 bases carry less
information than the start of V4: genus precision was 0.77 for V4-V5 and 0.38
for V3-V4. For short reads, the higher ranks are the safer summary.

## Confidence capping

The reported confidence at a rank is never higher than the confidence at the
rank above: species confidence is at most genus confidence, and so on. The
network predicts each rank separately, and the cap is applied when the table is
written. As a result, once one rank falls below its threshold, every rank below
it does too.

## Thresholds

When you give no threshold, a model that carries a calibrated operating point
uses its own cut-offs (all 26.11 and 26.11.1 models; the values are in
[models.md](models.md#operating-points)), and the run log says so. The
16S models (`v4`, `v3v4`, `v5v7`, `16s`, and since 26.11.1 `v4v5`) also carry a species gate, which withholds the species call
for reads far from the training records of the predicted species; it applies
with any thresholds, and only removes species calls.

A model without an operating point (the 26.10 files, the operon model, your own
models) uses 0.5 for domain to family, 0.7 for genus and 0.8 for species. Giving
`--confidence` or any `--<rank>-threshold` replaces the model's cut-offs with
this profile: `--confidence` sets the species threshold, and the other ranks
scale with it in the same ratio. `taxonova.classify()` in Python follows the
same rule (`confidence=` and the per-rank arguments):

| `--confidence` | domain to family | genus | species |
| --- | --- | --- | --- |
| 0.8 (default without an operating point) | 0.5 | 0.7 | 0.8 |
| 0.9 | 0.5625 | 0.7875 | 0.9 |
| 0.7 | 0.4375 | 0.6125 | 0.7 |
| 0.0 | 0 | 0 | 0 (every rank reported) |

A single rank can be set on its own, and then overrides the scaled value:

```bash
taxonova classify reads.fasta -m v4 --confidence 0.9 --genus-threshold 0.6
```

To reproduce the thresholds used in the paper's benchmark (0.5 for domain to
genus, 0.7 for species):

```bash
taxonova classify reads.fasta -m v4 --confidence 0.7 \
    --domain-threshold 0.5 --phylum-threshold 0.5 --class-threshold 0.5 \
    --order-threshold 0.5 --family-threshold 0.5 --genus-threshold 0.5
```

## Choosing a threshold

A higher threshold gives fewer species calls with fewer errors; a lower one
names more sequences, with more errors. The same threshold does not give the
same abstention rate for every region or reference. If a study needs a
particular balance, choose the threshold on sequences of known identity that
resemble your data, for example a mock community sequenced with the same
primers, or a held-out part of the reference.

To explore thresholds after a single run, classify with `--confidence 0` (every
rank is reported with its confidence, except species calls withheld by a
model's species gate) and apply cut-offs to the confidence columns afterwards. Apply them from domain downwards, so that a rank below its
cut-off also clears the ranks below it, as Taxonova does.

## Run log

Next to the table, `taxonova classify` writes a log with the package version,
the model file and the full command-line settings. Its name is the output name
with `.tsv` replaced by `.log`: `reads.taxonova.log` for the default output
`reads.taxonova.tsv`, and `results.log` for `-o results.tsv`. Keep it with the
results, or use `--no-log` to skip it. A run that stops before classification
starts (input not found, unknown model) does not write or replace a log.
