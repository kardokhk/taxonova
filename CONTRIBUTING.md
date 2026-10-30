# Contributing

Bug reports, questions and requests for new models are welcome as
[GitHub issues](https://github.com/kardokhk/taxonova/issues/new/choose).

## Reporting a problem

Please include the output of `taxonova --version`, the exact command, the model
and, if you can, a few sequences that show the problem.

## Changing the code

1. Fork the repository and create a branch from `main`.
2. Install in editable mode with the test dependencies:
   `pip install -e ".[test]"`.
3. Add or update tests in `tests/`, then run `python -m pytest`.
   Tests marked `models` need the released checkpoints and are skipped without
   them; set `TAXONOVA_TEST_MODELS_DIR` to a directory that holds them to run
   those too.
4. Describe the change in `CHANGELOG.md` under "Unreleased".
5. Open a pull request. The tests run automatically on Python 3.9 to 3.12.

A change that alters the classification output for an existing model needs a
note in the changelog that says how the output changes and why.

## Adding a model

See [docs/models.md](https://github.com/kardokhk/taxonova/blob/main/docs/models.md#adding-a-model).

## Releases

Maintainers follow [docs/releasing.md](https://github.com/kardokhk/taxonova/blob/main/docs/releasing.md).
