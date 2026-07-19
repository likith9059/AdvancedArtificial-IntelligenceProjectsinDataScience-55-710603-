# Week 3 Summary: Preprocessing and Baseline CNN

## Week 3 Aim
The aim of Week 3 was to build the first working model pipeline for casting defect detection. After completing Week 2 EDA and data-quality checks, the team moved into preprocessing, TensorFlow dataset creation, augmentation, class weighting and baseline CNN training.

## Work Completed
- Selected the unaugmented casting image dataset source.
- Created a clean image manifest.
- Created train, validation and test splits.
- Built a TensorFlow `tf.data` image-loading pipeline.
- Resized images to 224 × 224.
- Added training-only data augmentation.
- Calculated class weights to handle class imbalance.
- Built the first Custom CNN baseline model.
- Trained the baseline model with callbacks.
- Saved training curves, confusion matrix, classification report and baseline metadata.

## Member Contributions

| Name | Role | Week 3 Contribution |
|---|---|---|
| Sandeep Bommagoni | Data & EDA Lead | Clean manifest creation and train/validation/test split |
| Likith Kumar Devulapalli | Architecture / Xception Lead | TensorFlow preprocessing pipeline and augmentation preview |
| Raghava Krishna Battu | Model Comparison Lead | Custom CNN baseline model and training setup |
| Navatej Patel Cheerneni | Evaluation & XAI Lead | Baseline validation metrics, training curves and confusion matrix |
| Tharun Chinnachinnaiahgari | GitHub / Deployment Lead | File organisation, weekly summary and GitHub evidence tracking |

## Evidence Files
- `notebooks/Week_03_Preprocessing_and_Baseline_CNN.ipynb`
- `src/preprocessing_week3.py`
- `src/train_baseline_cnn.py`
- `reports/model_results/week3_split_counts.csv`
- `reports/model_results/week3_baseline_validation_metrics.json`
- `reports/model_results/week3_baseline_classification_report.csv`
- `reports/figures/week3_augmentation_preview.png`
- `reports/figures/week3_baseline_loss.png`
- `reports/figures/week3_baseline_accuracy.png`
- `reports/figures/week3_baseline_confusion_matrix.png`

## Week 3 Outcome
Week 3 produced the first measurable AI baseline. This baseline will be used in later weeks to compare whether Xception and other transfer-learning models provide a meaningful improvement.
