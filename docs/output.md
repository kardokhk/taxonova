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
such as `Prevotella sp002251435`.

The `taxonova classification` command, the Python function
`taxonova.classify` and the web app give the same table for the same model and
thresholds.

## Unclassified

At each rank, a label is reported only if its confidence reaches the threshold
for that rank. Otherwise the rank reads `Unclassified`, and so does every rank
below it. A sequence assigned to a genus but not to a species therefore has a
genus name and `Unclassified` at species. This is the model abstaining rather
than guessing, and it is expected for short regions in which related species
have identical sequences.

The confidence is printed even for `Unclassified` ranks, so you can see how
close a call was.

`Unknown` (rare) means the reference itself had no name at that rank for the
predicted lineage.

## Confidence capping

The reported confidence at a rank is never higher than the confidence at the
rank above: species confidence is at most genus confidence, and so on. The
network predicts each rank separately, and the cap is applied when the table is
written. As a result, once one rank falls below its threshold, every rank below
it does too.

## Thresholds

The default thresholds are 0.5 for domain to family, 0.7 for genus and 0.8 for
species. `--confidence` sets the species threshold, and the other ranks scale
with it in the same ratio:

| `--confidence` | domain to family | genus | species |
| --- | --- | --- | --- |
| 0.8 (default) | 0.5 | 0.7 | 0.8 |
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
rank is reported with its confidence) and apply cut-offs to the confidence
columns afterwards. Apply them from domain downwards, so that a rank below its
cut-off also clears the ranks below it, as Taxonova does.

## Run log

Next to the table, `taxonova classify` writes `<name>.taxonova.log` with the
package version, the model file and the full command-line settings. Keep it
with the results, or use `--no-log` to skip it.
