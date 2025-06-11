# Building the Documentation
eSLIM++ provides both, doxygen and sphinx documentation.
As the core functionality is completely developed in C++, the same description is used for both.

The current state of documentation is sparse, primarily the infrastructure and basic config files have been created.
It is limited to the docstrings provided by the code.

# Doxygen
To build the doxygen documentation (also required for the sphinx counter part), install doxygen first.
For example, on an Ubuntu system by
```bash
  sudo apt install doxygen
```

Then build the documentation in the `doc/doxygen` directory by
```bash
  doxygen Doxyfile
```
This will create a build directory, including the `index.html`, which can be opened by any browser.

# Sphinx
Before building the Sphinx documentation, build the doxygen part first.
Then, create a virtual python environment and install the requirements from `doc/sphinx/requirements.txt`, before building the documentation

```bash
  # select directory
  cd doc
  # create and use virtual env
  python3 -m venv venv
  source venv/bin/activate
  
  # install requirements
  pip install -r sphinx/requirements.txt
  
  #build documentation
  cd sphinx
  sphinx-build -M html source build
```

Afterward, a html directory is available at `doc/sphinx/build/html` including the index.html.

