#!/bin/bash
#
# ArchNeuronX Release Script
# 
# Usage:
#   ./scripts/release.sh [patch|minor|major] ["Release message"]
#
# Examples:
#   ./scripts/release.sh patch "Fix memory leak in CNN model"
#   ./scripts/release.sh minor "Add LSTM support"
#   ./scripts/release.sh major "Breaking API changes"

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

# Parse arguments
BUMP_TYPE="${1:-patch}"
RELEASE_MSG="${2:-Release $(date +%Y-%m-%d)}"

# Validate bump type
if [[ ! "$BUMP_TYPE" =~ ^(patch|minor|major)$ ]]; then
    echo -e "${RED}Error: Invalid bump type '$BUMP_TYPE'${NC}"
    echo "Usage: $0 [patch|minor|major] [\"Release message\"]"
    exit 1
fi

# Check for uncommitted changes
if ! git diff-index --quiet HEAD --; then
    echo -e "${YELLOW}Warning: You have uncommitted changes${NC}"
    git status --short
    echo ""
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Get current version
CURRENT_VERSION=$(cat "$PROJECT_ROOT/VERSION" 2>/dev/null || echo "1.0.0")
echo -e "${BLUE}Current version: ${CURRENT_VERSION}${NC}"

# Bump version using Python script
if command -v python3 &> /dev/null; then
    python3 "$SCRIPT_DIR/bump_version.py" --"$BUMP_TYPE" --dry-run
    echo ""
    read -p "Proceed with this version bump? (Y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Nn]$ ]]; then
        exit 1
    fi
    
    NEW_VERSION=$(python3 "$SCRIPT_DIR/bump_version.py" --"$BUMP_TYPE" 2>&1 | grep "New version:" | awk '{print $3}')
else
    # Manual version bump
    IFS='.' read -r MAJOR MINOR PATCH <<< "$CURRENT_VERSION"
    
    case "$BUMP_TYPE" in
        major)
            NEW_VERSION="$((MAJOR + 1)).0.0"
            ;;
        minor)
            NEW_VERSION="$MAJOR.$((MINOR + 1)).0"
            ;;
        patch)
            NEW_VERSION="$MAJOR.$MINOR.$((PATCH + 1))"
            ;;
    esac
    
    echo "$NEW_VERSION" > "$PROJECT_ROOT/VERSION"
    
    # Update CMakeLists.txt
    sed -i "s/project(ArchNeuronX VERSION .*)/project(ArchNeuronX VERSION $NEW_VERSION)/" \
        "$PROJECT_ROOT/CMakeLists.txt"
fi

echo -e "${GREEN}New version: ${NEW_VERSION}${NC}"

# Commit version bump
git add -A
git commit -m "chore(release): bump version to v${NEW_VERSION}" || echo "Nothing to commit"

# Create tag
TAG="v${NEW_VERSION}"
echo -e "${BLUE}Creating tag: ${TAG}${NC}"
git tag -a "$TAG" -m "Release ${NEW_VERSION}: ${RELEASE_MSG}"

# Push
echo ""
echo -e "${YELLOW}Ready to push. This will:${NC}"
echo "  1. Push commits to origin"
echo "  2. Push tag $TAG to origin"
echo "  3. Trigger GitHub Actions release workflow"
echo ""
read -p "Push to origin? (Y/n) " -n 1 -r
echo

if [[ ! $REPLY =~ ^[Nn]$ ]]; then
    git push origin main
    git push origin "$TAG"
    
    echo -e "${GREEN}✅ Release v${NEW_VERSION} pushed!${NC}"
    echo ""
    echo "📋 Release workflow: https://github.com/Gzeu/ArchNeuronX/actions/workflows/release.yml"
    echo "📦 Release page: https://github.com/Gzeu/ArchNeuronX/releases/tag/${TAG}"
else
    echo -e "${YELLOW}Release prepared locally but not pushed.${NC}"
    echo "To push later:"
    echo "  git push origin main"
    echo "  git push origin $TAG"
fi