# UKP_Psycontrol at SemEval-2026 Task 2: Modeling Valence and Arousal Dynamics from Text

This repository contains experimental software and is published for the sole purpose of giving additional background details on the respective publication. 

<details>
<summary>Abstract</summary>
This paper presents our system developed for SemEval-2026 Task 2. The task requires modeling both current affect and short-term affective change in chronologically ordered user-generated texts. We explore three complementary approaches: (1) LLM prompting under user-aware and user-agnostic settings, (2) a pairwise Maximum Entropy (MaxEnt) model with Ising-style interactions for structured transition modeling, and (3) a lightweight neural regression model incorporating recent affective trajectories and trainable user embeddings. Our findings indicate that LLMs effectively capture static affective signals from text, whereas short-term affective variation in this dataset is more strongly explained by recent numeric state trajectories than by textual semantics. Our system ranked first among participating teams in both Subtask 1 and Subtask 2A based on the official evaluation metric.
</details>

## Getting Started

This paper presents our system developed for **SemEval-2026 Task 2**.
The task requires modeling both current affect and short-term affective
change in chronologically ordered user-generated texts.

We explore three complementary approaches:

1.  **LLM prompting** under user-aware and user-agnostic settings\
2.  A **pairwise Maximum Entropy (MaxEnt) model** with Ising-style
    interactions for structured transition modeling\
3.  A **lightweight neural regression model** incorporating recent
    affective trajectories and trainable user embeddings

Our findings indicate that LLMs effectively capture static affective
signals from text, whereas short-term affective variation in this
dataset is more strongly explained by recent numeric state trajectories
than by textual semantics.

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

## Usage

### Using the classes

To import classes/methods of `semeval26_valence_arousal_from_text` from inside the package itself you can use relative imports: 

```py
from .base import BaseClass # Notice how I omit the package name

BaseClass().something()
```

To import classes/methods from outside the package (e.g. when you want to use the package in some other project) you can instead refer to the package name:

```py
from semeval26_valence_arousal_from_text import BaseClass # Notice how I omit the file name
from semeval26_valence_arousal_from_text.subpackage import SubPackageClass # Here it's necessary because it's a subpackage

BaseClass().something()
SubPackageClass().something()
```

### Using scripts

This is how you can use `semeval26_valence_arousal_from_text` from command line:

```bash
$ python -m semeval26_valence_arousal_from_text
```

### Expected results

After running the experiments, you should expect the following results:

(Feel free to describe your expected results here...)

### Parameter description

* `x, --xxxx`: This parameter does something nice

* ...

* `z, --zzzz`: This parameter does something even nicer

## Development

Read the FAQs in [ABOUT_THIS_TEMPLATE.md](ABOUT_THIS_TEMPLATE.md) to learn more about how this template works and where you should put your classes & methods. Make sure you've correctly installed `requirements-dev.txt` dependencies

## Cite

Please use the following citation:

```
@InProceedings{smith:20xx:CONFERENCE_TITLE,
  author    = {Smith, John},
  title     = {My Paper Title},
  booktitle = {Proceedings of the 20XX Conference on XXXX},
  month     = mmm,
  year      = {20xx},
  address   = {Gotham City, USA},
  publisher = {Association for XXX},
  pages     = {XXXX--XXXX},
  url       = {http://xxxx.xxx}
}
```

# Contact:
[UKP Lab](https://www.ukp.tu-darmstadt.de/) | [TU Darmstadt](https://www.tu-darmstadt.de/)

Don't hesitate to send us an e-mail or report an issue, if something is broken (and it shouldn't be) or if you have further questions.

