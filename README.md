# UKP_Psycontrol at SemEval-2026 Task 2: Modeling Valence and Arousal Dynamics from Text

This paper presents our system developed for **SemEval-2026 Task 2**.
The task requires modeling both current affect and short-term affective change in chronologically ordered user-generated texts.

We explore three complementary approaches:

1.  **LLM prompting** under user-aware and user-agnostic settings
2.  A **pairwise Maximum Entropy (MaxEnt) model** with Ising-style interactions for structured transition modeling
3.  A **lightweight neural regression model** incorporating recent affective trajectories and trainable user embeddings

Our findings indicate that LLMs effectively capture static affective signals from text, whereas short-term affective variation in this dataset is more strongly explained by recent numeric state trajectories than by textual semantics.

<details>
<summary>Abstract</summary>
    
>This paper presents our system developed for SemEval-2026 Task 2. 
>The task requires modeling both current affect and short-term affective change in chronologically ordered user-generated texts. 
>We explore three complementary approaches: (1) LLM prompting under user-aware and user-agnostic settings, (2) a pairwise Maximum Entropy (MaxEnt) model with Ising-style interactions for structured transition modeling, and (3) a lightweight neural regression model incorporating recent affective trajectories and trainable user embeddings. 
>Our findings indicate that LLMs effectively capture static affective signals from text, whereas short-term affective variation in this dataset is more strongly explained by recent numeric state trajectories than by textual semantics. 
>Our system ranked first among participating teams in both Subtask 1 and Subtask 2A based on the official evaluation metric.
</details>

## Usage

Detailed usage instructions are provided in the [semeval26_valence_arousal_from_text](semeval26_valence_arousal_from_text) directory.
Each subfolder contains documentation describing how to run the corresponding system:
```text
├── semeval26_valence_arousal_from_text   # main project directory
│   ├── llm_based_system                  # LLM-based solution for Subtask 1
│   ├── maxent                            # Maximum Entropy model for Subtasks 1 and 2A
│   └── neural_regression                 # Neural regression model for Subtask 2A
```

## SemEval-2026 Task 2 Description

**Task Name:** Predicting Variation in Emotional Valence and Arousal
over Time from Ecological Essays

The task focuses on modeling subjectively experienced emotion from
longitudinal, self-reported data.

The dataset contains chronologically ordered essays and feeling-word
lists written by U.S. service-industry workers over several years. Each
entry is paired with self-assessed:

-   **Valence (0-4)**
-   **Arousal (0-2)**

The shared task includes:

-   **Subtask 1:** Longitudinal Affect Assessment
    -   Predict valence and arousal per text
    -   Evaluated using Pearson correlation (r) and MAE
    -   Between-user, within-user, and composite evaluation
-   **Subtask 2A:** Forecasting Future Variation
    -   Predict next-step changes in valence and arousal
    -   Evaluated with user-level Pearson correlation and MAE
 
## Data Overview

-   2,764 entries from 137 users
-   Average 20 entries per user (median: 14)
-   52% free-form essays
-   48% feeling-word lists
-   Data span seven two-week periods
-   92% of users participated in only one period

The dataset used in this work is subject to the SemEval shared task usage agreement and therefore cannot be published or redistributed in this repository. The data may only be used for research purposes and must be obtained from the official task [website](https://semeval2026task2.github.io/SemEval-2026-Task2/overview).

## Contact
[Darya Hryhoryeva](mailto:darya.hryhoryeva@tu-darmstdadt.de) | [UKP Lab](https://www.ukp.tu-darmstadt.de/) | [TU Darmstadt](https://www.tu-darmstadt.de/)

Don't hesitate to send us an e-mail or report an issue, if something is broken (and it shouldn't be) or if you have further questions.

## Citation
If you found our data or code helpful, please cite our paper:
```text
@misc{hryhoryeva2026ukppsycontrolsemeval2026task2,
      title={UKP_Psycontrol at SemEval-2026 Task 2: Modeling Valence and Arousal Dynamics from Text}, 
      author={Darya Hryhoryeva and Amaia Zurinaga and Hamidreza Jamalabadi and Iryna Gurevych},
      year={2026},
      eprint={2604.21534},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2604.21534}, 
}
```

## License

This repository is licensed under the Apache License, Version 2.0. See LICENSE for the full license text.

## Disclaimer

>This repository contains experimental software and is published for the sole purpose of giving additional background details on the respective publication. 
