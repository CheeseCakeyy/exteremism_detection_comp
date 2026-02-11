# Exteremism Detection Project

## Overview:

This project explores a structured and disciplined approach to text
classification, progressing from simple probabilistic models to more
expressive linear classifiers while maintaining a strong focus on
correct evaluation practices.

The goal was not only to improve accuracy, but also to understand model
behavior, feature representations, and the risks of leaderboard-driven
optimization.

------------------------------------------------------------------------

## Methodology:

### 1. Naive Bayes: (Baseline)

-   Implemented a Naive Bayes classifier using **unigram features**
-   Established a fast, interpretable baseline
-   Served as a reference point for further improvements

### 2. Feature Enhancement with Bigrams:

-   Extended the feature space to include **bigram representations**
-   Captured short-range contextual information.
-   Observed measurable performance improvement over unigrams.

### 3. Logistic Regression:

-   Transitioned to Logistic Regression to relax independence
    assumptions.
-   Learned feature weights directly.
-   Tuned the regularization parameter (`C`) to analyze the
    bias--variance tradeoff.
-   Identified an optimal `C` value beyond which performance plateaued.

------------------------------------------------------------------------

## Evaluation Strategy:

-   Used **validation scores** as the primary indicator of model quality.
-   Compared validation performance with **public leaderboard scores.**
-   Observed that leaderboard gains were not always consistent with
    validation improvements.
-   Avoided overfitting to the public leaderboard subset.

------------------------------------------------------------------------
------------------------------------------------------------------------

##Public LB scores:
-    submissionNB.csv score = 0.794
-    submissionNB1.csv(after tuning NB) score = 0.816
-    submissionLR.csv score = 0.805

------------------------------------------------------------------------
## Results Summary:

  -----------------------------------------------------------------------
  Model        Features              Key Observation
  ------------ --------------------- ------------------------------------
  Naive Bayes  Unigrams              Strong baseline, fast and
                                     interpretable

  Naive Bayes  Bigrams               Improved contextual understanding

  Logistic     N-grams               Best overall performance, stable
  Regression                         generalization
  -----------------------------------------------------------------------

------------------------------------------------------------------------

## Conclusion:

This project demonstrates a complete machine learning workflow, starting
from baseline modeling and feature engineering to hyperparameter tuning
and evaluation awareness. The final Logistic Regression model achieved
competitive performance, after which further tuning resulted in
diminishing returns.

Rather than pursuing increasingly complex models for marginal accuracy
gains, the project prioritized **generalization, validation stability,
and methodological rigor**.

------------------------------------------------------------------------

## Key Takeaway:

*A well-tuned simple model, evaluated correctly, is often more valuable
than complex models optimized purely for leaderboard performance.*
