# Balancing Conflict and Harmony - Persistent Trust Revision in Time-Dependent Subjective Networks 

The scripts in this directory will create some of the images used in the paper with the title above.
More importantly, it also reproduces the evaluation results presented in the paper.
Evaluations can be run as descriped in [here](../README.md), for generating the images, some additional packages are required (`cm-super`,`texlive`, `texlive-latex-axtra`).

## Experiment description
To obtain the results of the first experiment run the script `symmetric_tr.py` >
It produces triangle similar to the paper and should be the only evaluation script requiring the packages listed above.
The second experiment is split into two parts:
Changing parameters and including the Baseline method is contained in `exp_nonconstant_paramter.py`, where some basic parameter are set in the first lines of code.
Feel free to change them and observe the impact in the results.
The observations of experiments by agents is part of `exp_distribution.py`, where the random experiment is run multiple times at each time step.
Respectively, the execution of the second script takes a little longer than the others, but a progress bar should estimate the approx. duration.

TLDR:
1) first experiment, single shot fusion, comparison of conflict und harmony based trust revision:
```bash
    exp_symTR.py
```
2) second experiment:
   1) first config, baseline method, change in parameters, observing single outcomes:
   ```bash
      exp_nonconstant_parameter.py
   ```
   2) second config, observing distributions / multiple outcomes
   ```bash
      exp_distribution.py
   ```

