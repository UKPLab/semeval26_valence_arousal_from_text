# semeval26_valence_arousal_from_text

This directory contains the implementations of all systems for **SemEval-2026 Task 2**.

Each subdirectory corresponds to one modeling approach described in the paper.  Please refer to the respective folder for detailed setup instructions.

## Available systems

- **`llm_based_system/`** > LLM prompting framework used for **Subtask 1**.

- **`maxent/`** > Pairwise Maximum Entropy (MaxEnt) model with structured transition modeling, used for **Subtasks 1 and 2A**.

- **`neural_regression/`**  > Lightweight neural regression model incorporating affective trajectories, used for **Subtask 2A**.


## Getting started

Navigate to the desired system directory and follow its local README:

```bash
cd semeval26_valence_arousal_from_text/<system_name>
