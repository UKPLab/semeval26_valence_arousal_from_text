# README

## Overview

These notebooks implement and run Maximum Entropy (MaxEnt) models for
two different subtasks:

-   `run_subtask1_MaxEnt.ipynb`
-   `run_subtask2a_MaxEnt.ipynb`


------------------------------------------------------------------------

## Project Structure

    .
    ├── run_subtask1_MaxEnt.ipynb
    ├── run_subtask2a_MaxEnt.ipynb
    ├── data/        # input datasets
    ├── models/      
    └── results/

------------------------------------------------------------------------

## How to Run

### Option 1: Run with Jupyter

``` bash
jupyter notebook
```

Open both notebooks and click:

Kernel → Restart & Run All

------------------------------------------------------------------------

### Option 2: Run from Command Line

``` bash
jupyter nbconvert --to notebook --execute run_subtask1_MaxEnt.ipynb
jupyter nbconvert --to notebook --execute run_subtask2a_MaxEnt.ipynb
```

------------------------------------------------------------------------

## Notes

-   Ensure all dataset paths inside notebooks are correct.

------------------------------------------------------------------------

## Troubleshooting

-   Install missing packages using `pip install package_name`
-   Verify dataset file paths
-   Restart kernel if execution fails
