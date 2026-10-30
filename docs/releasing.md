# Releasing (maintainers)

This is the checklist for every release. It is generalised from the 26.10.0
release. Order matters: the model files go up first because the package
downloads them, PyPI comes after GitHub because a PyPI version can never be
replaced, and the web app comes last because it installs the package from PyPI.

## Versions

- Package: calendar versioning `YY.MM.MICRO`. The first release in a month is
  `YY.MM.0`; further releases in the same month increase `MICRO`
  (26.10.0, 26.10.1, ...). The git tag is `v` plus the version.
- Models: each file carries its own model version (`taxonova-V4-greengenes2-26.10.pt`),
  set when the checkpoint is made. A released file never changes; see
  [models.md](models.md#versioning).
- Any change in classification output for an existing model is described in
  the changelog, with how the output changes and why.

## 1. Prepare the branch

1. Update the version in `pyproject.toml` and `taxonova/__init__.py`, and
   `version` and `date-released` in `CITATION.cff`.
2. Move the "Unreleased" changelog entries under a new heading
   `## [YY.MM.MICRO] - YYYY-MM-DD`.
3. If models are added or renamed: update `_REGISTRY` in `taxonova/models.py`
   (the release tag in `_RELEASE_TAG` if the files move to the new release),
   `docs/models.md`, and the README model table.
4. Run the full test suite with the checkpoints available:

   ```bash
   TAXONOVA_TEST_MODELS_DIR=/path/to/models python -m pytest
   ```

   The tests marked `models` must run, not skip, and pass.
5. Build and check locally:

   ```bash
   python -m build --outdir dist/
   python -m twine check --strict dist/*
   ```

   Install the wheel in a fresh environment and run `taxonova classify` on a
   test file with an empty models directory.

## 2. Model files (GitHub release and Hugging Face mirror)

Skip this step if no model file is new. Every file in `_REGISTRY` must be
downloadable from every URL it lists before the package is published.

1. Create the GitHub release for the tag and attach the model files and a
   `SHA256SUMS` file:

   ```bash
   gh release create vYY.MM.MICRO --title "taxonova YY.MM.MICRO" --notes-file notes.md \
       taxonova-*.pt SHA256SUMS
   ```

2. Upload the same files to the Hugging Face model repository
   `claessonlab/taxonova-models`, together with its model card (README.md):

   ```python
   from huggingface_hub import HfApi
   HfApi().upload_folder(repo_id="claessonlab/taxonova-models", repo_type="model",
                         folder_path="hf_models_repo", commit_message="Add ... models")
   ```

3. Download every file back from both addresses and check it with
   `sha256sum -c SHA256SUMS`.

Never delete or replace an asset of an earlier release: older package
versions download from there.

## 3. Tag and publish the package

1. Merge the branch into `main` and push.
2. Push the tag: `git tag -a vYY.MM.MICRO -m "taxonova YY.MM.MICRO" && git push origin vYY.MM.MICRO`.
3. The `release` workflow builds the wheel and sdist, attaches them to the
   GitHub release (creating it if step 2 was skipped), and publishes them to
   PyPI through Trusted Publishing. Watch it under the repository's Actions
   tab.

Without Trusted Publishing (or if the workflow fails), upload by hand with an
API token held in the environment, never in a file or a command:

```bash
TWINE_USERNAME=__token__ TWINE_PASSWORD="$PYPI_TOKEN" twine upload dist/*
```

### One-time setup of Trusted Publishing

On pypi.org, signed in as an owner of the `taxonova` project:

1. Open the project, then "Manage", then "Publishing".
2. Under "Add a new publisher", choose GitHub and enter: owner `kardokhk`,
   repository `taxonova`, workflow name `release.yml`, environment `pypi`.
3. In the GitHub repository settings, under "Environments", create an
   environment named `pypi` (optionally with required reviewers, so that each
   PyPI upload waits for approval).

After this, no PyPI token is needed. The workflow skips files that are already
on PyPI, so a version uploaded by hand does not make it fail.

## 4. Web app

The Hugging Face Space `claessonlab/Taxonova` installs a pinned version from
PyPI. After the package is on PyPI:

1. Change the `taxonova==` pin in the Space's requirements (and any version
   assertion in its `Dockerfile`) to the new version.
2. Push the change to the Space and wait for the build to reach "Running".
3. Classify a test file in the web app and compare it with the command line at
   the same threshold (rows that pass the web app's rRNA pre-filter must be
   identical).

## 5. After the release

- Install from PyPI in a clean environment, run `taxonova --version`,
  `taxonova models download <id>` with an empty models directory, and a
  classification compared with the output of step 1.
- Check that the README renders on GitHub and PyPI and that its links work.
- Start a new "Unreleased" section in the changelog.

## If something goes wrong

- GitHub release wrong, PyPI not yet published: delete the release and the
  tag, fix, and publish again.
- PyPI wrong: a version cannot be replaced. Yank it on pypi.org and release
  the next `MICRO` version.
- Web app wrong: revert the Space commit; the previous image serves until a
  build succeeds.
