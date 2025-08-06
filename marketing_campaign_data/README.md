# Bank Marketing

## Description

This project focuses on statistically rigorous and interpretable population profiling of [Bank Marktegin Dataset](https://archive.ics.uci.edu/dataset/222/bank+marketing). We focus on creating easy to interpret and descriptive profiles that maximize chances of subscribing, without hurting their size too much.

## Objectives

- Perform EDA and check assumptions for statistical tests and models
- Build interpretable models (logistic regression [glm], decision trees [rpart], LASSO [glmnet])
- Apply cross-validation to estimate generalization performance
- Evaluate model using confusion matrix, ROC curve, and metrics (accuracy, precision, F1, etc)
- Identify and validate high-performing customer segments (profiling)

## Project Structure
```
.
├── 01_eda.Rmd
├── 02_stat_tests.Rmd
├── 03_modeling.Rmd
├── 04_profiling.Rmd
├── master.Rmd       # Master notebook to render all four notebooks into single PDF file
├── data/            # Directory for raw data
├── models/          # Directory for models saved as RDS files
└── R/               # Directory for R scipts
```

All Rmarkdown files have their knitted pdf options on the same level as the notebooks are.

## Methods used

- **EDA vizualization:** Histograms, Mosaic plots, Box plots, Violin Plots, Q-Q plots
- **Statistical Tests:** t-test, Chi-squared, Anderson-Darling, ANOVA, Kruskal-Wallis, Spearman correlation
- **Models:** Logistic regression (`glm`), Decision trees (`rpart`)
- **Feature Engineering:** One-hot encoding, greedy feature filtering driven by AIC, LASSO
- **Model Evaluation:** Cross-validation, ROC, accuracy, precision, F1 score
- **Profiling:** Score segmentation, proportion tests, binomial test

## Reproduction

To render all notebooks run `R/render.r` script. This can be done from command line from root of this project.

```{bash}
Rscript R/render.R
```

### Requirements

- R (≥ 4.0)
- R packages:
  - `caret`, `rpart`, `glmnet`,
  - `pROC`, `corrplot`, `car`
  - `knitr`, `rmarkdown`
- pandoc (≥ 1.3)
