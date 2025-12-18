# Release Instructions

This document describes how to create a new release of c2i2o.

## Prerequisites

- Maintainer access to the GitHub repository
- PyPI account with permissions for c2i2o project


## Release Process

### 1. Prepare Release

```bash
# Run the release preparation script
./scripts/prepare_release.sh 0.1.0
```

# This will:
# - Run all tests
# - Run pre-commit checks
# - Build documentation
# - Run examples
# - Update version numbers
# - Build and check package


### 2. Update CHANGELOG

Edit CHANGELOG.md:

    Move items from [Unreleased] to new version section
    Add release date
    Ensure all changes are documented

### 3. Commit Version Bump

```bash
    git add pyproject.toml docs/source/conf.py CHANGELOG.md
    git commit -m "chore: bump version to 0.1.0"
    git push origin main
```

### 4. Create Git Tag

```bash
    git tag -a v0.1.0 -m "Release version 0.1.0"
    git push origin v0.1.0
```

This triggers the pre-release checks workflow.

### 5. Create GitHub Release

    Go to https://github.com/KIPAC/c2i2o/releases/new
    Select tag: v0.1.0
    Release title: v0.1.0 - Initial Release
    Description: Copy from CHANGELOG.md
    Check "Set as the latest release"
    Click "Publish release"

This automatically triggers the PyPI publishing workflow!


### 6. Verify Release

```bash
    # Wait a few minutes for GitHub Actions to complete

    # Check PyPI
    open https://pypi.org/project/c2i2o/

    # Test installation

    python -m venv test_release
    source test_release/bin/activate
    pip install c2i2o==0.1.0
    python -c "import c2i2o; print(c2i2o.__version__)"
    deactivate
    rm -rf test_release
```

### 7. Post-Release Tasks

```bash

    # Bump to next dev version
    ./scripts/post_release.sh 0.1.0

    # Commit
    git add pyproject.toml CHANGELOG.md
    git commit -m "chore: bump version to 0.2.0-dev"
    git push origin main
```

### 8. Announce Release

    Update documentation website


## Testing on Test PyPI

To test the release process without publishing to PyPI:
Option 1: Manual Workflow Trigger

    1. Go to https://github.com/KIPAC/c2i2o/actions/workflows/publish.yml
    2. Click "Run workflow"
    3. Select branch: main
    4. Click "Run workflow"

This publishes to Test PyPI.
Option 2: Create Pre-Release

    1. Create a pre-release on GitHub with tag like v0.1.0-rc1
    2. Mark as "pre-release"
    2. Mark as "pre-release"
    3. Manually trigger Test PyPI upload from Actions

### Verify Test PyPI Installation

```bash
    # Create test environment
    python -m venv testpypi_test
    source testpypi_test/bin/activate

    # Install from Test PyPI
    pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple c2i2o

    # Test
    python -c "import c2i2o; print(c2i2o.__version__)"

    # Clean up
    deactivate
    rm -rf testpypi_test
```
