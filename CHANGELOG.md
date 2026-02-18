# Changelog

All notable changes to ArchNeuronX will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Automated release pipeline with GitHub Actions
- Binary builds for Linux and Windows
- Docker multi-platform support (amd64/arm64)
- Semantic versioning automation

## [1.0.0] - 2025-10-04

### Added
- Initial release of ArchNeuronX
- Neural network trading system with MLP and CNN architectures
- LibTorch 2.1.0 integration for C++ neural networks
- CUDA support for GPU acceleration
- REST API for trading bot integration
- Real-time data processing for Crypto & Forex
- Docker containerization
- CMake build system with cross-platform support
- Unit testing with Google Test
- Doxygen documentation generation

### Technical Details
- C++17 compatible
- CMake 3.18+ required
- Supports GCC 9+, Clang 10+, MSVC 2019+
- Optional CUDA 11.8+ for GPU support

---

## Release Process

### Automated Releases

Releases are automatically created when pushing tags:

```bash
# Create and push a new version tag
git tag v1.2.0
git push origin v1.2.0
```

Or trigger manually via GitHub Actions workflow dispatch.

### Version Bumping

Use the bump_version script:

```bash
# Analyze commits and auto-bump
python scripts/bump_version.py

# Force specific bump
python scripts/bump_version.py --major  # Breaking changes
python scripts/bump_version.py --minor  # New features
python scripts/bump_version.py --patch  # Bug fixes

# Preview without changes
python scripts/bump_version.py --dry-run
```

### Conventional Commits

This project follows [Conventional Commits](https://www.conventionalcommits.org/):

- `feat:` → Minor version bump (new features)
- `fix:` → Patch version bump (bug fixes)
- `BREAKING CHANGE:` or `feat!:` → Major version bump
- `docs:`, `style:`, `refactor:`, `test:`, `chore:` → No version bump

---

## Version History

| Version | Date | Highlights |
|---------|------|------------|
| 1.0.0 | 2025-10-04 | Initial release |
| | | Neural network trading system |
| | | CUDA support |
| | | REST API integration |