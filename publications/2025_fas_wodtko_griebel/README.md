# A Unified Self-Assessment Framework for Autonomous Driving Stacks Using Subjective Logic

This directory contains scripts used to generate figures and reproduce the evaluation results presented in the paper  
**“A Unified Self-Assessment Framework for Autonomous Driving Stacks Using Subjective Logic.”**

📌 *Evaluations can be run as described in the main [README](../README.md).*

---

## 🛠️ Required Packages for Figure Generation

To generate the figures shown in the paper, make sure the following LaTeX-related packages are installed:

- `cm-super`
- `texlive`
- `texlive-latex-extra`
- `dvipng`


## 🧪 Experiment Description

To reproduce the evaluation results for the Lucid System Architecture, run:

```bash
python3 evaluate_lucid_system_architectur.py
```

This script evaluates both self-assessment formulations introduced in the paper:
- ✅ Overall State of Health
- 🚨 Safety-Critical Check

It outputs the performance and reliability metrics for each formulation, as discussed in the publication.
