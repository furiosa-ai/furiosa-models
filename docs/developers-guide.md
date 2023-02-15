# Developer's guide

This documentation is for developers who want to contribute to Furiosa Models.

## Release guide

### Preparing for Release

- [ ] Select a correct release tag to mark the release: `x.y.z`
- [ ] Update the code in the `main` branch to reflect the next development version.
    - [ ] `__version__` field in `furiosa/models/__init__.py`
- [ ] Create a dedicated release branch on github for the new tag.

### Pre-Release Tasks

Before releasing the new version, ensure that the following tasks are completed:

- [ ] Update the code in the release branch to the appropriate version.
- [ ] Generate a rendered documentation for the new release.
- [ ] Write a changelog that describes the changes made in the new release.
- [ ] Test the building of wheels by `flit build` to ensure the release is functional.

### Releasing the New Version

- [ ] Publish the package to PyPI with `flit publish` command. 🎉
