# Week 4 Summary: Xception Transfer Learning

## Week 4 Aim
The aim of Week 4 was to train the main Xception transfer-learning model for casting defect detection and compare its validation performance against the Week 3 Custom CNN baseline.

## Work Completed
- Reused the clean dataset preparation process from Week 3.
- Created a stratified train/validation/test split.
- Calculated class weights.
- Built a TensorFlow input pipeline using 299 × 299 images.
- Built an Xception transfer-learning model.
- Loaded ImageNet-pretrained Xception weights where available.
- Froze the Xception backbone and trained the custom classification head.
- Added callbacks for checkpointing, early stopping and learning-rate reduction.
- Saved training curves, validation metrics, confusion matrix and classification report.
- Prepared model metadata and comparison structure against the Week 3 baseline.

## Member Contributions

| Name | Role | Week 4 Contribution |
|---|---|---|
| Sandeep Bommagoni | Data & EDA Lead | Verified dataset source, split consistency and class weights |
| Likith Kumar Devulapalli | Architecture / Xception Lead | Built the Xception transfer-learning architecture |
| Raghava Krishna Battu | Model Comparison Lead | Added callbacks, checkpointing and comparison with baseline |
| Navatej Patel Cheerneni | Evaluation & XAI Lead | Evaluated Xception using validation metrics and confusion matrix |
| Tharun Chinnachinnaiahgari | GitHub / Deployment Lead | Saved metadata, summary, meeting notes and GitHub evidence |

## Week 4 Outcome
Week 4 produced the first transfer-learning model. This prepares the project for Week 5 model comparison, where Xception will be compared against other architectures such as MobileNetV2 and EfficientNetB0.
