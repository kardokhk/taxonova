# GitHub Upload Guide for Taxonova

This guide will help you replace the old version of Taxonova on GitHub with this new version.

## Overview

You will be replacing the entire content of your existing repository at https://github.com/kardokhk/taxonova with this new version. The .gitignore file has been configured to exclude unnecessary files.

## Files That Will Be Uploaded

The following essential files will be uploaded to GitHub:

### Core Package Files
- `taxonova/` - Main Python package directory with all source code
- `setup.py` - Package installation configuration
- `pyproject.toml` - Modern Python project configuration
- `requirements.txt` - Package dependencies
- `MANIFEST.in` - Controls which files are included in distribution

### Documentation
- `README.md` - Main documentation (newly created, academic style)
- `LICENSE` - MIT License
- `CHANGELOG.md` - Version history
- `QUICKSTART.md` - Quick reference guide

## Files That Will NOT Be Uploaded

The .gitignore file excludes:
- `test_*.py` - Local test files
- `build_and_test.sh` - Local testing script
- `database/*` - Large FASTA files (users download separately)
- `models/*` - Large model files (users download via pip)
- `input/*` - Test input data
- `output/*` - Test output data
- `bin/taxonova` - Compiled/generated script
- `dist/`, `build/`, `*.egg-info/` - Build artifacts
- Internal documentation files (`IMPLEMENTATION_SUMMARY.md`, `READY_TO_PUBLISH.md`, `PUBLISHING.md`, `MODEL_HOSTING.md`)

## Step-by-Step Upload Instructions

### Option 1: Replace Everything (Recommended)

This is the cleanest approach - it completely replaces the old repository with the new version.

```bash
# 1. Navigate to your taxonova directory
cd /Users/kardokhkakabra/Downloads/taxonova

# 2. Initialize git (if not already initialized)
git init

# 3. Add the remote repository (if not already added)
git remote add origin https://github.com/kardokhk/taxonova.git

# If the remote already exists, update it:
# git remote set-url origin https://github.com/kardokhk/taxonova.git

# 4. Create a new branch for the update
git checkout -b main

# 5. Add all files (respecting .gitignore)
git add .

# 6. Check what will be committed (verify no unwanted files)
git status

# 7. Commit the changes
git commit -m "Release v25.11.0: Complete rewrite with hierarchical prototypical classification

- Implemented hierarchical prototypical neural network architecture
- Added pre-trained models for multiple 16S rRNA regions
- Integrated model download system via pip
- Added smart full-data training with two-stage approach
- Published to PyPI (https://pypi.org/project/taxonova/)
- Added web interface at HuggingFace Spaces
- Complete documentation overhaul
- MIT License"

# 8. Force push to replace the old version
git push -f origin main

# Note: -f flag is needed to overwrite the old repository history
# This is acceptable since you want to completely replace the old version
```

### Option 2: Preserve Git History

If you want to keep the old commit history:

```bash
# 1. Navigate to your taxonova directory
cd /Users/kardokhkakabra/Downloads/taxonova

# 2. Clone your existing repository to a temporary location
cd ..
git clone https://github.com/kardokhk/taxonova.git taxonova-old
cd taxonova-old

# 3. Remove all old files except .git directory
find . -not -path './.git/*' -not -name '.git' -delete

# 4. Copy new files from your working directory
cp -r ../taxonova/* .
cp ../taxonova/.gitignore .

# 5. Add and commit
git add .
git status  # Verify the changes
git commit -m "Release v25.11.0: Complete rewrite with hierarchical prototypical classification"

# 6. Push to GitHub
git push origin main
```

### Option 3: Using GitHub Desktop or Web Interface

1. **Backup the old repository** (optional but recommended):
   - Go to https://github.com/kardokhk/taxonova
   - Click "Code" → "Download ZIP" to save a backup

2. **Delete all files in the GitHub repository** (via web):
   - Navigate to the repository on GitHub
   - You can delete files individually or use Git commands

3. **Upload new files**:
   - Use GitHub's web upload feature
   - Or use GitHub Desktop application

## After Upload Checklist

Once you've pushed to GitHub:

- [ ] Visit https://github.com/kardokhk/taxonova to verify the upload
- [ ] Check that README.md displays correctly on the main page
- [ ] Verify that unwanted files (test files, models, databases) are NOT present
- [ ] Create a new release (optional but recommended):
  ```bash
  # Create and push a tag
  git tag -a v25.11.0 -m "Release version 25.11.0"
  git push origin v25.11.0
  ```
- [ ] On GitHub, go to "Releases" → "Create a new release"
  - Choose tag: v25.11.0
  - Release title: "Taxonova v25.11.0"
  - Description: Copy from CHANGELOG.md
- [ ] Update repository description on GitHub to:
  "Deep learning-based taxonomic classification for bacterial 16S rRNA amplicon sequences"
- [ ] Add topics/tags: `bioinformatics`, `taxonomy`, `deep-learning`, `16s-rrna`, `microbiome`, `metagenomics`

## Verify Everything Works

After uploading, test that users can install and use your package:

```bash
# Install from PyPI
pip install taxonova

# Download models
taxonova-setup

# Test classification
taxonova classification --help
```

## Troubleshooting

### Authentication Issues

If you encounter authentication problems:

```bash
# Use personal access token instead of password
# Go to GitHub Settings → Developer Settings → Personal Access Tokens
# Generate a new token with 'repo' permissions
# Use the token as your password when prompted
```

### Files Not Being Ignored

If unwanted files are being added:

```bash
# Remove cached files from git
git rm -r --cached .
git add .
git commit -m "Fix .gitignore"
```

### Large Files Warning

If GitHub warns about large files:
- Ensure `models/*.pt` and `database/*.fasta` are in .gitignore
- These files should NOT be uploaded to GitHub
- Users download them separately via `taxonova-setup`

## Contact

If you encounter issues:
- Check GitHub documentation: https://docs.github.com/
- Review git basics: https://git-scm.com/doc

---

**Ready to upload!** The repository is configured and ready for GitHub.
