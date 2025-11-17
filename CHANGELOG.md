# Changelog

All notable changes to Taxonova will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [25.11.0] - 2025-11-17

### Added
- Initial PyPI release
- Pre-trained model download system
  - Interactive model selection
  - CLI command `taxonova download-models`
  - Support for multiple V regions (V1-V9, V3-V4, V4, full rRNA)
  - Models stored in `~/.taxonova/models/` by default
- Post-installation setup command `taxonova-setup`
- Model path resolution from user directory
- Configuration system for model locations
- Environment variable support (`TAXONOVA_MODELS_DIR`, `TAXONOVA_CONFIG_DIR`)
- Comprehensive documentation
  - PyPI publishing guide (PUBLISHING.md)
  - Updated README with installation instructions
  - Model download documentation
- MIT License
- .gitignore for Python projects
- Installation test script

### Changed
- Package structure optimized for PyPI distribution
- Entry points using console_scripts instead of scripts
- Model loading now supports automatic path resolution
- MANIFEST.in updated to exclude local models and databases
- Enhanced setup.py with complete metadata for PyPI
- Improved pyproject.toml with project URLs and keywords

### Fixed
- Model path handling for both absolute and relative paths
- Classification mode now provides helpful error messages when models are not found

## [Previous versions]

Previous development versions (pre-PyPI) not tracked in changelog.

---

## Version Numbering Scheme

Taxonova uses a date-based versioning scheme: `YY.MM.PATCH`

- `YY`: Two-digit year
- `MM`: Two-digit month
- `PATCH`: Patch number (incremented for releases within the same month)

Example: 25.11.0 = November 2025, first release
