# Developer's Guide

This documentation is for developers who want to contribute to Furiosa Models.

## Prepare the Docker image for the CI

The CI for this project depends on an (external) Docker image:

`asia-northeast3-docker.pkg.dev/next-gen-infra/furiosa-ai/furiosa-models:base`

You can create the Docker image by running `make docker-build`,
push the image to the Docker registry by running `make docker-push`.

## Release guide

### Preparing for Release

- [ ] Select a correct release tag to mark the release: `x.y.z`
- [ ] Update the code in the `main` branch to reflect the next development version.
    - [ ] `__version__` field in `furiosa/models/__init__.py`
- [ ] Create a dedicated release branch on github for the new tag.

### Pre-Release Tasks

Before releasing the new version, ensure that the following tasks are completed:

- [ ] Update the code in the release branch to the appropriate version.
- [ ] Generate a rendered documentation using `/release-doc/v1.2.3` command for the new release.
- [ ] Write a changelog that describes the changes made in the new release.
- [ ] Test the building of wheels by `flit build` to ensure the release is functional.

### Releasing the New Version

- [ ] Publish the package to PyPI with `flit publish` command. 🎉
