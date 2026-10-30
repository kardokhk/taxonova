# Releasing (maintainers)

This is the checklist for every release. It is generalised from the 26.10.0
release. Order matters. The tag is made from the merged `main`, so it points
at the commit that carries the new version. PyPI comes last of the three
package steps, after both model addresses work, because a PyPI version can
never be replaced. The web app comes after PyPI because it installs the
package from there.

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
3. If models are added: add them to `_REGISTRY` in `taxonova/models.py` with
   `urls=_urls(filename, tag="vYY.MM.MICRO")`, where the tag is this release
   (the one whose GitHub release will carry the file). Leave the tags of
   existing models unchanged. Update `docs/models.md` and the README model
   table.
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

## 2. Tag the merged commit

1. Merge the branch into `main` and push `main`.
2. Tag the merge commit and push the tag:

   ```bash
   git checkout main && git pull
   git tag -a vYY.MM.MICRO -m "taxonova YY.MM.MICRO"
   git push origin vYY.MM.MICRO
   ```

   Do not create the release with `gh release create` before this point.
   If the tag does not exist yet, `gh` makes it from the current default
   branch, which may be the commit before the merge.
3. The push starts the `release` workflow. Its `build` job checks that the tag
   matches the package version and builds the wheel and sdist. Its
   `github-release` job creates the GitHub release and attaches them. Its
   `pypi` job waits for approval (see the one-time setup below). Do not
   approve it yet.

## 3. Model files (GitHub release and Hugging Face mirror)

Skip this section if no model file is new. Every file in `_REGISTRY` must be
downloadable from every URL it lists before the package reaches PyPI.

1. Put the new model files, under their registry names, in an empty folder
   together with the model card (`README.md` of the Hugging Face model
   repository) and a checksum file. Copy the files; do not include any
   working notes or manifests that name local paths.

   ```bash
   cd release-models/
   sha256sum taxonova-*.pt > SHA256SUMS
   ```

2. When the `github-release` job has finished, attach the files to the
   release it created:

   ```bash
   gh release upload vYY.MM.MICRO taxonova-*.pt SHA256SUMS
   ```

3. Upload the same folder to the Hugging Face model repository
   `claessonlab/taxonova-models`:

   ```python
   from huggingface_hub import HfApi
   HfApi().upload_folder(repo_id="claessonlab/taxonova-models", repo_type="model",
                         folder_path="release-models", commit_message="Add ... models")
   ```

4. In another empty folder, download every file back from both addresses
   (the URLs in `_REGISTRY`) and check it with `sha256sum -c SHA256SUMS`.

Never delete or replace an asset of an earlier release: older package
versions download from there.

## 4. Publish the package to PyPI

1. When section 3 is complete, approve the waiting `pypi` job of the
   `release` workflow (Actions tab, the run for the tag, "Review deployments").
   It publishes the wheel and sdist through Trusted Publishing.
2. If no model file is new, approve it as soon as the `github-release` job has
   succeeded.

Without Trusted Publishing (or if the workflow fails), upload by hand with an
API token held in the environment, never in a file or a command. Name the
wheel and sdist explicitly, so that no other file in `dist/` is sent:

```bash
TWINE_USERNAME=__token__ TWINE_PASSWORD="$PYPI_TOKEN" \
    twine upload dist/taxonova-YY.MM.MICRO-py3-none-any.whl dist/taxonova-YY.MM.MICRO.tar.gz
```

### One-time setup of Trusted Publishing

On pypi.org, signed in as an owner of the `taxonova` project:

1. Open the project, then "Manage", then "Publishing".
2. Under "Add a new publisher", choose GitHub and enter: owner `kardokhk`,
   repository `taxonova`, workflow name `release.yml`, environment `pypi`.
3. In the GitHub repository settings, under "Environments", create an
   environment named `pypi` and add yourself under "Required reviewers". This
   is required, not optional: it is what makes the PyPI upload wait until the
   model files are online (section 4).

After this, no PyPI token is needed. The workflow skips files that are already
on PyPI, so a version uploaded by hand does not make it fail.

## 5. Web app

The Hugging Face Space `claessonlab/Taxonova` installs a pinned version from
PyPI. After the package is on PyPI:

1. Change the `taxonova==` pin in the Space's requirements (and any version
   assertion in its `Dockerfile`) to the new version.
2. Push the change to the Space and wait for the build to reach "Running".
3. Classify a test file in the web app and compare it with the command line at
   the same threshold (rows that pass the web app's rRNA pre-filter must be
   identical).

## 6. After the release

- Install from PyPI in a clean environment, run `taxonova --version`,
  `taxonova models download <id>` with an empty models directory, and a
  classification compared with the output of section 1.
- Check that the README renders on GitHub and PyPI and that its links work.
- Start a new "Unreleased" section in the changelog.

## If something goes wrong

- GitHub release wrong, PyPI not yet published: reject the waiting `pypi`
  job, delete the release and the tag (`gh release delete vYY.MM.MICRO
  --cleanup-tag`), fix, and start again from section 2.
- PyPI wrong: a version cannot be replaced. Yank it on pypi.org and release
  the next `MICRO` version.
- Web app wrong: revert the Space commit; the previous image serves until a
  build succeeds.
