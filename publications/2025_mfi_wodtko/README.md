# Balancing Conflict and Harmony - Persistent Trust Revision in Time-Dependent Subjective Networks 

The scripts in this directory will create some of the images used in the paper with the title above.
More importantly, it also reproduces the evaluation results presented in the paper.
Evaluations can be run as described in [here](../README.md).

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

# Citation
Please use the following citation for reference to the publication in this directory (`2025_mfi_wodtko_balancing.bib`):
```bibtex
@INPROCEEDINGS{wodtko2025balancing,
  author={Wodtko, Thomas and Buchholz, Michael},
  booktitle={2025 IEEE International Conference on Multisensor Fusion and Integration for Intelligent Systems (MFI)},
  title={Balancing Conflict and Harmony - Persistent Trust Revision in Time-Dependent Subjective Networks},
  year={2025},
  volume={},
  number={},
  pages={1-8},
  keywords={Monte Carlo methods;Uncertainty;Estimation;Sensor systems and applications;Random variables;Reliability;Logic;Intelligent systems},
  doi={10.1109/MFI67357.2025.11259128}
}
```
