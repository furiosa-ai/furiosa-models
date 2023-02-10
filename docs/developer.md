# Developer's guide

This documentation is for developers who want to contribute to Furiosa Models.

## Release guide

### Select the appropriate commit

First, you need to select the proper release candidate commit to release.
That revision *must pass* all CI tests and be polished.

### Create a dedicated release branch

Create a release branch named `branch-<release version w/o v prefix>` from that commit.

### Set the version

Furiosa Models' main branch is for developers, so it set to `<next-release-version>.dev0`.
The version information is located in `./furiosa/models/__init__.py`. Please set it with the appropriate one.

### Alter absolute links in documentation

Since `README.md` can appear in many places, including GitHub, the rendered documentation, and the index page of PyPI, there are a few absolute URL links (e.g., `[Getting Started](https://furiosa-ai.github.io/furiosa-models/v0.8.0/getting_started/)`).
Please change it to appropriate links with the proper version.

### Create release tag

Create a git tag when you publish a release to keep track of revisions in your releases.
You can create and push a git tag with the following commands.

```shell
git tag <release version>  # e.g. git tag v0.8.2
git push --tags <appropriate remote>  # e.g. git push --tags upstream
```

### Test building wheels

We're using `flit` as our packaging tool, so please test that you can generate a proper wheel using the `flit build` command before releasing.
You may want to install `flit` first, please install it with `pip install flit`.

## Publish to PyPI

Now you can publish the package with `flit publish` command. :tada:
