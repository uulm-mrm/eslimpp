# Evaluations run with eSLIM++

generally, python evaluations can be run using the eSLIM++ python interface providing the `subjective_logic ` package.
For this, simply install eSLIM++ in a virtual env and run the desired python evaluation script.
The following provides an example for a specific script of the eSLIM++ publication:

The description below assumes to be at a location where a venv can be generated, and the eSLIM++ git repository is explicitly cloned.\
If the repository is already available and a venv with eslimpp installed was created, simply switch to the `eslimpp/publications` directory and follow the descript below installing the eval specific requirements and running the eval scripts.

```bash
  # install eslimpp in a virtual environment with arbitrary location
  python3 -m venv venv
  source venv/bin/activate
  pip install eslimpp
  
  # clone directory to get evaluation scripts
  # only if not previously cloned or downloaded
  git clone git@github.com:uulm-mrm/eslimpp.git
  cd eslimpp/publications
  
  # install evaluation specific requirements
  pip install -r eSLIM++/requirements.txt
  
  # run desired evaluation script
  python eSLIM++/classification_task.py
```

