#!/usr/bin/env python3
"""
Semantic Version Bumper for ArchNeuronX
Automatically determines version bump based on conventional commits.

Usage:
    python scripts/bump_version.py [--major|--minor|--patch] [--dry-run]

Conventional Commit Types:
    feat:     → MINOR bump (new feature)
    fix:      → PATCH bump (bug fix)
    BREAKING: → MAJOR bump (breaking change)
    docs,style,refactor,test,chore: → no bump (but updates changelog)
"""

import re
import subprocess
import sys
from pathlib import Path
from typing import Tuple, Optional

VERSION_FILE = Path(__file__).parent.parent / "VERSION"
CMAKE_FILE = Path(__file__).parent.parent / "CMakeLists.txt"


def get_current_version() -> Tuple[int, int, int]:
    """Read current version from VERSION file or CMakeLists.txt."""
    
    if VERSION_FILE.exists():
        content = VERSION_FILE.read_text().strip()
        match = re.search(r'(\d+)\.(\d+)\.(\d+)', content)
        if match:
            return tuple(int(x) for x in match.groups())
    
    if CMAKE_FILE.exists():
        content = CMAKE_FILE.read_text()
        match = re.search(r'project\(\w+\s+VERSION\s+(\d+)\.(\d+)\.(\d+)', content)
        if match:
            return tuple(int(x) for x in match.groups())
    
    return (1, 0, 0)


def get_commits_since_last_tag() -> list:
    """Get all commits since the last tag."""
    try:
        # Get last tag
        result = subprocess.run(
            ["git", "describe", "--tags", "--abbrev=0"],
            capture_output=True, text=True, check=True
        )
        last_tag = result.stdout.strip()
        
        # Get commits since tag
        result = subprocess.run(
            ["git", "log", f"{last_tag}..HEAD", "--pretty=format:%s"],
            capture_output=True, text=True, check=True
        )
        return result.stdout.strip().split('\n') if result.stdout.strip() else []
    except subprocess.CalledProcessError:
        # No tags yet, get all commits
        result = subprocess.run(
            ["git", "log", "--pretty=format:%s", "--no-merges"],
            capture_output=True, text=True, check=True
        )
        return result.stdout.strip().split('\n') if result.stdout.strip() else []


def analyze_commits(commits: list) -> Tuple[bool, bool, bool]:
    """
    Analyze commits to determine version bump.
    Returns: (major_bump, minor_bump, patch_bump)
    """
    major = minor = patch = False
    
    for commit in commits:
        commit_lower = commit.lower()
        
        # Breaking changes → MAJOR
        if 'breaking' in commit_lower or '!' in commit.split(':')[0]:
            major = True
        
        # New features → MINOR
        if commit_lower.startswith('feat:') or commit_lower.startswith('feat('):
            minor = True
        
        # Bug fixes → PATCH
        if commit_lower.startswith('fix:') or commit_lower.startswith('fix('):
            patch = True
    
    return major, minor, patch


def bump_version(current: Tuple[int, int, int], major: bool, minor: bool, patch: bool,
                 force_major: bool = False, force_minor: bool = False, force_patch: bool = False) -> Tuple[int, int, int]:
    """Calculate new version number."""
    major_val, minor_val, patch_val = current
    
    if force_major or major:
        return (major_val + 1, 0, 0)
    elif force_minor or minor:
        return (major_val, minor_val + 1, 0)
    elif force_patch or patch:
        return (major_val, minor_val, patch_val + 1)
    else:
        # Default to patch bump
        return (major_val, minor_val, patch_val + 1)


def update_version_files(version: Tuple[int, int, int], dry_run: bool = False) -> None:
    """Update VERSION and CMakeLists.txt with new version."""
    version_str = f"{version[0]}.{version[1]}.{version[2]}"
    
    if dry_run:
        print(f"[DRY RUN] Would update VERSION to: {version_str}")
        print(f"[DRY RUN] Would update CMakeLists.txt to: project(ArchNeuronX VERSION {version_str})")
        return
    
    # Update VERSION file
    VERSION_FILE.write_text(f"{version_str}\n")
    print(f"✅ Updated VERSION to: {version_str}")
    
    # Update CMakeLists.txt
    if CMAKE_FILE.exists():
        content = CMAKE_FILE.read_text()
        new_content = re.sub(
            r'project\(\w+\s+VERSION\s+\d+\.\d+\.\d+',
            f'project(ArchNeuronX VERSION {version_str}',
            content
        )
        CMAKE_FILE.write_text(new_content)
        print(f"✅ Updated CMakeLists.txt to: {version_str}")


def generate_changelog(version: Tuple[int, int, int], commits: list) -> str:
    """Generate changelog from commits."""
    version_str = f"{version[0]}.{version[1]}.{version[2]}"
    
    features = []
    fixes = []
    breaking = []
    other = []
    
    for commit in commits:
        commit_lower = commit.lower()
        
        if 'breaking' in commit_lower or '!' in commit.split(':')[0]:
            breaking.append(commit)
        elif commit_lower.startswith('feat:') or commit_lower.startswith('feat('):
            features.append(commit)
        elif commit_lower.startswith('fix:') or commit_lower.startswith('fix('):
            fixes.append(commit)
        else:
            other.append(commit)
    
    changelog = [f"## v{version_str}\n"]
    
    if breaking:
        changelog.append("### 💥 Breaking Changes\n")
        for c in breaking:
            changelog.append(f"- {c}")
        changelog.append("")
    
    if features:
        changelog.append("### ✨ Features\n")
        for c in features:
            changelog.append(f"- {c}")
        changelog.append("")
    
    if fixes:
        changelog.append("### 🐛 Bug Fixes\n")
        for c in fixes:
            changelog.append(f"- {c}")
        changelog.append("")
    
    if other:
        changelog.append("### 📝 Other Changes\n")
        for c in other[:10]:  # Limit to 10
            changelog.append(f"- {c}")
        changelog.append("")
    
    return '\n'.join(changelog)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Bump version based on conventional commits')
    parser.add_argument('--major', action='store_true', help='Force major bump')
    parser.add_argument('--minor', action='store_true', help='Force minor bump')
    parser.add_argument('--patch', action='store_true', help='Force patch bump')
    parser.add_argument('--dry-run', action='store_true', help='Show what would happen without making changes')
    parser.add_argument('--changelog', action='store_true', help='Generate changelog only')
    
    args = parser.parse_args()
    
    current = get_current_version()
    print(f"📌 Current version: {current[0]}.{current[1]}.{current[2]}")
    
    commits = get_commits_since_last_tag()
    print(f"📝 Found {len(commits)} commits since last tag")
    
    if args.changelog:
        changelog = generate_changelog(current, commits)
        print(changelog)
        return
    
    if not commits:
        print("⚠️  No commits found since last tag. Use --patch to force bump.")
        if not args.patch:
            return
    
    major, minor, patch = analyze_commits(commits)
    
    print(f"\n📊 Analysis:")
    print(f"   Breaking changes: {'✅' if major else '❌'}")
    print(f"   New features: {'✅' if minor else '❌'}")
    print(f"   Bug fixes: {'✅' if patch else '❌'}")
    
    new_version = bump_version(
        current, major, minor, patch,
        force_major=args.major,
        force_minor=args.minor,
        force_patch=args.patch
    )
    
    print(f"\n🔢 New version: {new_version[0]}.{new_version[1]}.{new_version[2]}")
    
    update_version_files(new_version, dry_run=args.dry_run)
    
    if not args.dry_run:
        changelog = generate_changelog(new_version, commits)
        print(f"\n📋 Changelog:\n")
        print(changelog)


if __name__ == "__main__":
    main()