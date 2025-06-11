# Building pypi package for distribution

to create a new package and upload it to (test)pypi, the following steps are required:
Serves as reminder

```bash
    # for reasons, the venv must be outside the eslim++ dir
    python3 -m venv venv
    source venv/bin/activate
    # install nanobind and scikit-build-core is needed
    python -m pip install build twine auditwheel
    # build for current linux dist
    cd [path_to_eslim++]
    python -m build 
    # fix build to manylinux, which is accepted by (test)pypi
    # example: auditwheel repair dist/eslimpp-0.1.1-cp312-abi3-linux_x86_64.whl -w dist/
    auditwheel repair dist/eslimpp-[LIB_VERSION]-[CURRENT SYSTEM INFO].whl -w dist/
    # remove old wheel
    rm dist/*-linux*x86_64.whl
    # upload to testpypi (change for true distr), there might be a request for an access token
    python -m twine upload --repository testpypi dist/*
```

If file already exists, a version change is required...

