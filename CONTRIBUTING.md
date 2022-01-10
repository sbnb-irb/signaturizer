# Contributing

## Adding features or fixing bugs

* Fork the repo
* Check out a feature or bug branch
* Add your changes
* Update README when needed
* Submit a pull request to upstream repo
* Add description of your changes
* Ensure tests are passing
* Ensure branch is mergeable

## Testing

* Please make sure tests pass with `./script/test`

## Release a new package version

Publication of the package on PyPI is automated in the CI pipeline, however bumping the version and creating a release tag (that triggers publication) is manual and should be performed as follows:

Be sure that all unit tests are passing.

Select the new version number. Consider that it is not possible to re-publish with the same version nor it is possible to reduce it.

Bump (e.g. 1.0.1 -> 1.0.2 or 1.1.0) the version number in the following files:

* [signaturizer/__init__.py](http://gitlabsbnb.irbbarcelona.org/packages/signaturizer/-/blob/master/signaturizer/__init__.py)
* [setup.py](http://gitlabsbnb.irbbarcelona.org/packages/signaturizer/-/blob/master/setup.py)

Push these changes.

Create a release tag and push it:

```bash
git tag v1.0.2
git push origin v1.0.2
```

This will trigger CI pipeline to publish the package officially (and definetively) on (PyPI)[https://pypi.org/project/signaturizer/#history]
