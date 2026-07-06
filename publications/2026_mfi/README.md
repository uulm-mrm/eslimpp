# Sequential Information Fusion in Subjective Logic: Advancing Efficient Temporal Evidence Filtering
This directory contains scripts to reproduce the results of the publication *Sequential Information Fusion in Subjective Logic: Advancing Efficient Temporal Evidence Filtering* at MFI 2026.
The publication was submitted and is currently under review and thus, not publicly available yet.
Further, the scripts contain further tests and evaluations extending the results of the publication and validating the presented SL operators.

## Experiment Description
The following describes the available experiments, demonstrates the script usage, explains the presented output and relates them to the publication.
Refer to the `README.md` in the publications directory for installation instructions.

### Sequential Fusion Operators
The `seq_fusion_operators.py` is used to verify the correct implementation of the proposed sequential operators by comparing their results to Multi-Source Fusion.
Results are reported for Average Belief Fusion (ABF) and Weighted Belief Fusion (WBF) on 100 random opinions, 1000 random runs and the Josang example opinions.
The expected result is that the sequential operators match the result of multi-source fusion, while the naive sequential fusion does not obtain the correct result.
This script is also used to export the results on the Josang example opinions, which are shown in Table 1 in the publication.
The seed for the RNG initialization and the number of runs can be modified in the script. 
```
python3 seq_fusion_operators.py
```

### Sequential Fusion Opinion Space
After verifying that our sequential operators match the Multi-Source Fusion results, we verify that the formulation in the evidence-space matches the presented formulation in the opinion-space in the `seq_fusion_opinion_space.py` script.
For that, we consider different opinions dimension and the ABF / WBF.
We perform the sequential fusion of 25 random SL opinions and compare the result in evidence- and opinion-space.
`apply_discount=True` can be used to test the discounted operation, which was not possible for the Multi-Source Fusion comparison.
```
python3 seq_fusion_opinion_space.py
```

### Temporal Evidence Filter (TEF)
The `tef.py` script visualizes the behavior of the TEF on different sequences.
Using the helper functions in `util.py` a number of sequences for evaluation are generated.
The script processes the sequence and plots the behavior of the TEF.
The top plots shows the projected probability of the input opinion and TEF output in blue and orange respectively, and their corresponding uncertainty as a dashed line.
The bottom plot shows the result of the Monte-Carlo simulation and the degree of conflict in blue and orange respectively, and their corresponding threshold as a dashed line.
The colored bar above the plot shows the different processes, green refers to `GOOD_OPINION`, orange a transition and red to `BAD_OPINION`.
The script is used to create and export the data used for Fig. 4/5/6 in the publication.
Fig. 4 uses scenario `transition_jump` while Fig. 5/6 use `single_jump`.
New scenarios can be created by using the helper functions.
The parameters of the TEF and the internal conflict handling can be set by the parameters in the script.
```
python3 tef.py
```
### Reset Strategy
The `reset_strategy.py` script performs 10000 random runs on jump sequences, evaluating the proposed reset strategies.
The parameters of the TEF, the fusion type and the desired noise levels can be set through the parameters inside the script.
The script compares the detected change point in the ST memory to the ground-truth and classifies each as correct, early or late.
This corresponds to the results shown in Table 2.
```
python3 reset_strategy.py
```

### Runtimes
The `runtimes.py` script reports the runtime of the ST memory reset strategies, the Multi-Source Fusion operation and sequential fusion.
This substantiates the claims made in the paper about the runtime of the algorithm
```
python3 runtimes.py
```

