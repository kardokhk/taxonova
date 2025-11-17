#!/bin/bash
# Quick Commands for GitHub Upload
# Make this file executable: chmod +x github_upload_commands.sh

echo "=========================================="
echo "Taxonova GitHub Upload - Quick Commands"
echo "=========================================="
echo ""

# Navigate to directory
echo "Step 1: Navigate to taxonova directory"
echo "cd /Users/kardokhkakabra/Downloads/taxonova"
echo ""

# Initialize git
echo "Step 2: Initialize git (if needed)"
echo "git init"
echo ""

# Add remote
echo "Step 3: Set up remote repository"
echo "git remote add origin https://github.com/kardokhk/taxonova.git"
echo "# OR if remote already exists:"
echo "git remote set-url origin https://github.com/kardokhk/taxonova.git"
echo ""

# Create main branch
echo "Step 4: Create/switch to main branch"
echo "git checkout -b main"
echo ""

# Add files
echo "Step 5: Add all files (respects .gitignore)"
echo "git add ."
echo ""

# Check status
echo "Step 6: Review what will be committed"
echo "git status"
echo ""
echo "IMPORTANT: Verify that these files are NOT listed:"
echo "  - test_*.py"
echo "  - models/*.pt"
echo "  - database/*.fasta"
echo "  - build/, dist/, *.egg-info/"
echo ""

# Commit
echo "Step 7: Commit the changes"
cat << 'EOF'
git commit -m "Release v25.11.0: Complete rewrite with hierarchical prototypical classification

- Implemented hierarchical prototypical neural network architecture
- Added pre-trained models for multiple 16S rRNA regions
- Integrated model download system via pip
- Added smart full-data training with two-stage approach
- Published to PyPI (https://pypi.org/project/taxonova/)
- Added web interface at HuggingFace Spaces
- Complete documentation overhaul
- MIT License"
EOF
echo ""

# Push
echo "Step 8: Push to GitHub (force push to replace old version)"
echo "git push -f origin main"
echo ""

# Create tag
echo "Step 9: Create version tag"
echo "git tag -a v25.11.0 -m 'Release version 25.11.0'"
echo "git push origin v25.11.0"
echo ""

echo "=========================================="
echo "After Upload:"
echo "=========================================="
echo "1. Visit https://github.com/kardokhk/taxonova"
echo "2. Verify README.md displays correctly"
echo "3. Check that unwanted files are NOT present"
echo "4. Create a release:"
echo "   - Go to Releases → Create a new release"
echo "   - Tag: v25.11.0"
echo "   - Title: 'Taxonova v25.11.0'"
echo "   - Copy description from CHANGELOG.md"
echo "5. Update repository settings:"
echo "   - Description: 'Deep learning-based taxonomic classification for bacterial 16S rRNA amplicon sequences'"
echo "   - Website: https://huggingface.co/spaces/claessonlab/Taxonova"
echo "   - Topics: bioinformatics, taxonomy, deep-learning, 16s-rrna, microbiome, metagenomics"
echo ""
echo "=========================================="
echo "Done!"
echo "=========================================="
