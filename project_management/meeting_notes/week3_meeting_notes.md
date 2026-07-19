# Week 3 Meeting Notes

## Meeting Topic
Preprocessing and Baseline CNN Development

## Attendees
Sandeep Bommagoni, Likith Kumar Devulapalli, Raghava Krishna Battu, Navatej Patel Cheerneni, Tharun Chinnachinnaiahgari

## Completed This Week
1. The team moved from EDA into modelling preparation.
2. A clean image manifest was prepared for model input.
3. Train, validation and test splits were created.
4. A TensorFlow `tf.data` pipeline was built.
5. Image resizing and batching were tested.
6. Data augmentation was added for training images.
7. Class weights were calculated.
8. A Custom CNN baseline was implemented and trained.
9. Baseline results, curves and confusion matrix were saved.

## Key Discussion
The team agreed that the baseline CNN is necessary because later transfer-learning models must be compared against a simple model. The group also agreed that evaluation should not rely only on accuracy; defect recall, precision, F1-score, ROC-AUC and PR-AUC should also be monitored.

## Blockers
Training time may vary depending on Kaggle GPU availability. If GPU is unavailable, the notebook may run slowly.

## Next Week Plan
Week 4 will focus on training the main Xception transfer-learning model and comparing its validation performance against the Week 3 Custom CNN baseline.
