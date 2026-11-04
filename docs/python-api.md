# Python interface

## Classify

```python
import taxonova

df = taxonova.classify("reads.fasta", model="v4")
```

`df` is a pandas DataFrame with the columns of the command-line output
(`sequence_id`, then a label and a `<rank>_confidence` column for each rank
from domain to species). Labels are strings and confidences are floats with the
four decimals printed in the table. See [output.md](output.md).

The input can be:

- a FASTA path (plain or `.gz`);
- FASTA text, starting with `>`;
- a dict `{id: sequence}`;
- a list of `(id, sequence)` pairs;
- a list of sequences, which are named `seq1`, `seq2`, ...

```python
df = taxonova.classify({"asv1": "TACGGAGGGTGCAAGCGTTAATCGG...", "asv2": "TACGTAGGGGGC..."},
                       model="v3v4", confidence=0.7, threads=8)
species_counts = df.loc[df["species"] != "Unclassified", "species"].value_counts()
```

Arguments:

| argument | default | meaning |
| --- | --- | --- |
| `model` | `"v4"` | model id, earlier name, or path to a `.pt` file |
| `confidence` | `None` | species threshold; the other ranks scale with it. `None`: the model's calibrated operating point, or 0.8 for a model without one, as on the command line |
| `threads` | `None` (2) | CPU threads |
| `domain_threshold` ... `genus_threshold` | `None` | override the scaled threshold of one rank (and, like `confidence`, replace the operating point) |
| `models_dir` | `None` | where models are stored (default as on the command line) |
| `quiet` | `False` | suppress download messages |
| `strand` | `"plus"` | `"both"` classifies each read in the orientation the model is more confident about |

Thresholds follow `taxonova classify`: with no `confidence` and no per-rank
threshold, a model with a calibrated operating point uses its own cut-offs
(see [models.md](models.md#operating-points)) and a model without one uses 0.8
with the scaled profile. Any threshold you pass replaces the operating point;
the species gate of a model that has one applies either way. Before 26.11.0, `confidence` defaulted to
0.8; pass `confidence=0.8` to keep that behaviour.

The model is downloaded on first use. Each call loads the model again, so for
many small batches it is faster to pass all sequences in one call.

## Models

`taxonova.models` is the registry the command line and the web app use.

```python
from taxonova.models import list_models, get_model, ensure_model, models_dir

for spec in list_models():
    print(spec.id, spec.region, spec.reference, spec.filename)

spec = get_model("V3-V4")        # ids, earlier names and file names; case-insensitive
path = ensure_model("v3v4")      # pathlib.Path to a verified local copy; downloads once
```

| function | returns |
| --- | --- |
| `list_models(include_hidden=False)` | list of `ModelSpec` in display order; preview and deprecated models only with `include_hidden=True` |
| `get_model(name)` | the `ModelSpec`; raises `KeyError` that lists the valid ids |
| `ensure_model(name_or_path, models_dir=None, quiet=False)` | path to a local copy, downloaded and verified if needed |
| `models_dir()` | `$TAXONOVA_MODELS_DIR`, else `~/.taxonova/models` |

A `ModelSpec` has the fields `id`, `name`, `region`, `reference`,
`reference_version`, `taxonomy`, `model_version`, `filename`, `urls`, `sha256`,
`md5`, `size_mb`, `status` (`"stable"`, `"preview"` or `"deprecated"`),
`aliases` and `description`.

These names are kept stable across releases. Code written for taxonova 26.10.0
that imports `taxonova.utils.model_downloader` (`AVAILABLE_MODELS`,
`ModelDownloader`) keeps working; new code should use `taxonova.models`.
