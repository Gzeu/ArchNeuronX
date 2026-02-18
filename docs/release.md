# ArchNeuronX Release Guide

## 🚀 Quick Release

### Option 1: Using the Release Script (Recommended)

```bash
# Patch release (bug fixes)
./scripts/release.sh patch "Fix memory leak in CNN"

# Minor release (new features)
./scripts/release.sh minor "Add LSTM layer support"

# Major release (breaking changes)
./scripts/release.sh major "Redesigned API"
```

### Option 2: Manual Tag Push

```bash
# Update version
echo "1.2.0" > VERSION

# Commit and tag
git add VERSION CMakeLists.txt
git commit -m "chore: bump version to 1.2.0"
git tag v1.2.0

# Push to trigger release
git push origin main
git push origin v1.2.0
```

### Option 3: GitHub Actions UI

1. Go to **Actions** → **Release Pipeline**
2. Click **Run workflow**
3. Enter version number (e.g., `1.2.0`)
4. Check **Pre-release** if needed
5. Click **Run workflow**

---

## 📦 What Gets Released

### Binaries
- **Linux x64**: `archneuronx-{version}-linux-x64.tar.gz`
- **Windows x64**: `archneuronx-{version}-win-x64.zip`

### Docker Images
```bash
docker pull ghcr.io/gzeu/archneuronx:latest
docker pull ghcr.io/gzeu/archneuronx:1.2.0
docker pull ghcr.io/gzeu/archneuronx:1.2
```

### GitHub Release
- Release notes with changelog
- Binary downloads
- Installation instructions

---

## 📝 Conventional Commits

Version bumps are determined by commit messages:

| Type | Version Bump | Example |
|------|--------------|---------|
| `feat:` | Minor (0.1.0) | `feat: add LSTM support` |
| `fix:` | Patch (0.0.1) | `fix: memory leak in CNN` |
| `BREAKING` | Major (1.0.0) | `feat!: redesigned API` |
| `docs:`, `chore:` | None | `docs: update README` |

### Breaking Changes

Include `BREAKING CHANGE:` in commit body or add `!` after type:

```bash
feat!: redesigned neural network API

BREAKING CHANGE: The `train()` method now requires a config object
instead of individual parameters.
```

---

## 🔧 Version Bump Script

```bash
# Analyze commits and suggest version
python scripts/bump_version.py

# Force specific bump type
python scripts/bump_version.py --major
python scripts/bump_version.py --minor
python scripts/bump_version.py --patch

# Preview without changes
python scripts/bump_version.py --dry-run

# Generate changelog only
python scripts/bump_version.py --changelog
```

---

## 🔄 Release Pipeline Stages

```
┌─────────────────┐
│  prepare-release │ → Determine version, generate changelog│
└────────┬────────┘│
         │
    ┌────┴────┬─────────────┐
    │         │             │
    ▼         ▼             ▼
┌────────┐ ┌─────────┐ ┌──────────────┐
│ Linux  │ │ Windows │ │    Docker    │
│ Build  │ │  Build  │ │ Multi-arch   │
└────┬───┘ └────┬────┘ └──────┬───────┘
     │          │              │
     └──────────┴──────────────┘│
                │
                ▼
        ┌───────────────┐
        │ create-release│ → GitHub release with assets
        └───────┬───────┘
                │
                ▼
          ┌─────────┐
          │ notify  │ → Webhook notifications
          └─────────┘
```

---

## ⚠️ Pre-Release Checklist

Before releasing, ensure:

- [ ] All tests pass (`ctest --output-on-failure`)
- [ ] Code is formatted (`make format`)
- [ ] Documentation is updated
- [ ] CHANGELOG.md has been updated
- [ ] No breaking changes without major version bump
- [ ] Docker builds locally (`docker build -t test .`)

---

## 🐛 Troubleshooting

### Build Fails

```bash
# Check LibTorch is downloaded
ls libtorch/

# Rebuild clean
rm -rf build && mkdir build && cd build
cmake .. -DCMAKE_PREFIX_PATH=$PWD/../libtorch
make -j$(nproc)
```

### Docker Push Fails

- Check GitHub token has `packages:write` permission
- Verify `GITHUB_TOKEN` secret is set

### Version Not Updated

```bash
# Manually update VERSION file
echo "1.2.0" > VERSION

# Update CMakeLists.txt
sed -i 's/project(ArchNeuronX VERSION .*)/project(ArchNeuronX VERSION 1.2.0)/' CMakeLists.txt
```

---

## 📊 Release Monitoring

After release:

1. **Check GitHub Actions**: https://github.com/Gzeu/ArchNeuronX/actions
2. **Verify Release Page**: https://github.com/Gzeu/ArchNeuronX/releases
3. **Test Docker Pull**: `docker pull ghcr.io/gzeu/archneuronx:latest`
4. **Test Binary Download**: Download from release page and run

---

## 🔐 Required Permissions

The workflow requires these permissions:

- `contents: write` - Create releases and tags
- `packages: write` - Push Docker images to GHCR

These are automatically granted to `GITHUB_TOKEN` in the workflow.