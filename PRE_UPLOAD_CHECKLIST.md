# Pre-Upload Verification Checklist

Complete this checklist before uploading to GitHub.

## ✅ Files Ready

- [x] README.md - Academic, professional, comprehensive
- [x] .gitignore - Excludes unnecessary files
- [x] LICENSE - MIT License present
- [x] setup.py - Package configuration
- [x] pyproject.toml - Modern Python config
- [x] requirements.txt - Dependencies listed
- [x] CHANGELOG.md - Version history
- [x] MANIFEST.in - Distribution control
- [x] taxonova/ - Complete source code

## ✅ Files Excluded

- [x] test_*.py - Not needed on GitHub
- [x] build_and_test.sh - Local testing only
- [x] database/*.fasta - Too large, users download separately
- [x] models/*.pt - Too large, users download via pip
- [x] input/ - Test data
- [x] output/ - Test output
- [x] bin/ - Generated scripts
- [x] dist/, build/, *.egg-info/ - Build artifacts
- [x] IMPLEMENTATION_SUMMARY.md - Internal docs
- [x] READY_TO_PUBLISH.md - Internal docs
- [x] PUBLISHING.md - Internal docs
- [x] MODEL_HOSTING.md - Internal docs
- [x] .pypirc.template - PyPI credentials template

## ✅ README Quality Check

Content verified:
- [x] Professional, academic tone
- [x] Clear description of what Taxonova does
- [x] Web interface prominently featured
- [x] pip installation instructions
- [x] Pre-trained models documentation
- [x] Usage examples for classification
- [x] Usage examples for training
- [x] Input/output format specifications
- [x] Performance metrics
- [x] Citation information
- [x] License information
- [x] Support/contact information
- [x] Acknowledgments

Format verified:
- [x] Badges at top (Python, License, PyPI)
- [x] Table of contents
- [x] Code blocks properly formatted
- [x] Tables properly formatted
- [x] Links work correctly
- [x] No TODO or placeholder text

## ✅ Package Configuration

- [x] setup.py has correct version (25.11.0)
- [x] pyproject.toml has correct version (25.11.0)
- [x] Author name: Kardokh Kaka Bra
- [x] Email: kardokhkaka@gmail.com
- [x] GitHub URL: https://github.com/kardokhk/taxonova
- [x] License: MIT
- [x] Python version: >=3.8

## 🔍 Pre-Upload Commands

Run these commands to verify everything is correct:

```bash
cd /Users/kardokhkakabra/Downloads/taxonova

# 1. Check for large files that shouldn't be uploaded
find . -type f -size +10M -not -path '*/\.*' -not -path '*/models/*' -not -path '*/database/*'
# Should show NO results or only expected files

# 2. Preview what will be committed
git add -n .
# Review the list - should NOT include test files, models, databases

# 3. Check README renders correctly
head -50 README.md
# Verify it looks professional

# 4. Verify .gitignore is working
git status --ignored
# Should show models/, database/, test files as ignored
```

## 📋 Upload Steps

Once everything is verified:

1. **Initialize and add remote**:
   ```bash
   cd /Users/kardokhkakabra/Downloads/taxonova
   git init
   git remote add origin https://github.com/kardokhk/taxonova.git
   ```

2. **Add and commit**:
   ```bash
   git checkout -b main
   git add .
   git status  # Final review
   git commit -m "Release v25.11.0: Complete rewrite with hierarchical prototypical classification"
   ```

3. **Push to GitHub**:
   ```bash
   git push -f origin main
   git tag -a v25.11.0 -m "Release version 25.11.0"
   git push origin v25.11.0
   ```

4. **Verify on GitHub**:
   - Visit https://github.com/kardokhk/taxonova
   - Check README displays correctly
   - Verify no unwanted files present
   - Create a release from the v25.11.0 tag

5. **Update repository settings**:
   - Description: "Deep learning-based taxonomic classification for bacterial 16S rRNA amplicon sequences"
   - Website: https://huggingface.co/spaces/claessonlab/Taxonova
   - Topics: bioinformatics, taxonomy, deep-learning, 16s-rrna, microbiome, metagenomics

## ⚠️ Important Notes

- The `-f` flag in `git push -f` will overwrite the old repository
- This is intentional since you want to replace the old version completely
- Make sure you have a backup of the old version if needed
- Large files (models, databases) will be downloaded by users via `taxonova-setup`

## ✅ Post-Upload Verification

After uploading:

1. **Test installation**:
   ```bash
   pip install taxonova
   taxonova --help
   ```

2. **Test model download**:
   ```bash
   taxonova-setup
   ```

3. **Check GitHub**:
   - README displays correctly
   - All necessary files present
   - No unwanted files uploaded
   - License shows as MIT
   - Topics/tags added

## 🎉 Ready to Upload!

All checks passed. You're ready to upload Taxonova to GitHub!

For detailed instructions, see:
- GITHUB_UPLOAD_GUIDE.md
- github_upload_commands.sh (executable reference)
- PREPARATION_SUMMARY.md (overview of what was done)
