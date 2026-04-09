# Self-Assessment in Autonomous Driving Perception Systems 

This directory contains scripts that reproduce the results of some experiments in my dissertation (Thomas Wodtko).
The Thesis is submitted and is currently being reviewed, thus, not yet publicly available.
Reviewers on any kind ("Gutachter"/"Prüfer") and all others are highly welcomed to view the provided scripts and run the evaluations.

## Information with respect to the thesis
As mentioned, the script in this directory reproduce some of my results.
More specifically, they reproduce experiments from Chapter 2, dealing with subjective-logic-based reasoning in general, and Chapter 3, considering self-assessment.
Evaluations of Chapter 4 are not within this library, as they consider the grid map application.
With all scripts, parameters that are described in the thesis are set towards the beginning of the script.
They can be changed, but their validity is explicitly not verified, hence, changes to the parameters must be reasonable and senseful.


# Experiment Description
In the following, the scripts of this directory are linked to the location in the thesis.

## Chapter 2
Experiments of chapter 2 are:

### Dirichlet
For the illustration in Figure 2.2 in Section 2.1.2, opinions are generated and converted to Dirichlet distributions.
The Dirichlet evidence parameter can be obtained using the script:
```bash
    beta.py
```

### Trusted Fusion and Trust Revision
Considering the example for trust discount and trusted fusion from Section 2.1.7, 
the example for trust revision of Section 2.1.8, 
and the experiments to the proposed trust revision operators two scripts are available.
First, in 
```bash
    tr_josang_cs.py
```
the trust revision approach from Jøsang is compared to the proposed conflict-shares-based trust revision (cf. Figure 2.11 in Section 2.5.1).
This script also includes the basic trusted fusion without trust revision.
Next, in
```bash
    tr_cs_hscs.py
```
the conflict-shares-based trust revision is compared to the combined harmony- and conflict-based trust revision (cf. again Figure 2.11 in Section 2.5.1).

### Time-Dependent Trust Revision
The time series evaluation of Section 2.5.1 "Time-Dependent Trust Revision" is available with
```bash
    time_dependent_weather_forecast.py
```
Here, the progression of the projected probability is plotted with the ground truth in one plot, and a second plot shows the errors for each time step.

### Reliability Estimation 
The reliability estimation evaluation of Section 2.5.2 is split into "Observing Single Events" and "Observing Distributions".
Both the scripts below require the package `alive_progress` to be installed for python.
For the first, the evaluation is available with
```bash
    exp_nonconstant_parameter.py
```
It runs multiple Monte Carlo iterations (default 5000) and shows the progress while running.
Then the results from Figure 2.14 are illustrated.
By setting the parameter
```python3
    # agents can see all outcomes
    test_uncertain = 0.0
    # agents miss 30% of outcomes
    test_uncertain = 0.3
```
the visibility of agents is contolled.\
For the second part "Observing Distributions", the evaluation is available with
```bash
    exp_distribution.py
```
In contrast to the thesis's evaluations, only 5 Monte Carlo iterations are calculated to reduce wait time.
By changing the parameter for trust revision, i.e., 
```python3
    # last entry is the used weight
    weighted_types_cs_avg = [
        # first setting (conflict weight higher than harmonty weight)
        sl.WeightedTypes(sl.RelationType.CONFLICT, sl.TrustRevisionType.REFERENCE_FUSION, sl.ConflictType.BELIEF_AVERAGE,0.75),
        sl.WeightedTypes(sl.RelationType.HARMONY, sl.TrustRevisionType.REFERENCE_FUSION, sl.ConflictType.BELIEF_AVERAGE, 0.25),
        
        # second setting (both equally weighted with 0.5)
        # sl.WeightedTypes(sl.RelationType.CONFLICT, sl.TrustRevisionType.REFERENCE_FUSION, sl.ConflictType.BELIEF_AVERAGE,0.5),
        # sl.WeightedTypes(sl.RelationType.HARMONY, sl.TrustRevisionType.REFERENCE_FUSION, sl.ConflictType.BELIEF_AVERAGE,0.5),
    ]
```
The two described behaviors are obtained (cf. Section 2.5.2 "Observing Distributions")

## Chapter 3
Experiments of chapter 3 include:

### Unified Self-Assessment Framework
The evaluation of the self-assessment framework from Section 3.3.4 is available with
```bash
    evaluate_lucid_system_architecture.py
```
Using the values specified in Equations (3.49),(3.50), and (3.51), configured as shows in Figure 3.13 (a),
the script yields the results of Figure 3.13 (a)&(b)

