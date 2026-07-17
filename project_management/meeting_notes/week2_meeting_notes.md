# Week 2 Meeting Notes

## Meeting Topic
EDA and Data Quality Analysis

## Attendees
Sandeep Bommagoni, Likith Kumar Devulapalli, Raghava Krishna Battu, Navatej Patel Cheerneni, Tharun Chinnachinnaiahgari

## Completed This Week
1. Dataset folder structure was confirmed.
2. Image inventory dataframe was created.
3. Class distribution was analysed.
4. Image dimension and channel metadata were inspected.
5. Random class samples were visualised.
6. Corrupted image check was added.
7. Duplicate image hash check was added.
8. Average image and pixel intensity analysis were added.
9. Week 2 notebook and weekly summary were prepared.

## Key Discussion
The team agreed that EDA must be completed before training. Duplicate-image checking is especially important because exact duplicates can inflate test performance if they appear across train and test folders.

## Blockers
No major blockers. Dataset must be available in the Kaggle path before running the notebook.

## Next Week Plan
Week 3 will build the preprocessing pipeline, apply augmentation, calculate class weights and train a baseline CNN.
