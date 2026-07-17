# Week 2 Summary: EDA and Data Quality

## Week 2 Aim
The aim of Week 2 was to understand the casting image dataset before model training.

## Work Completed
- Loaded dataset folder paths.
- Created image inventory dataframe.
- Analysed class distribution.
- Checked image readability.
- Analysed image dimensions and channels.
- Displayed sample images from both classes.
- Checked corrupted/unreadable files.
- Checked duplicate images using MD5 hashing.
- Generated average images by class.
- Compared grayscale pixel intensity distributions.
- Saved EDA outputs and results.

## Member Contributions

| Name | Role | Contribution |
|---|---|---|
| Sandeep Bommagoni | Data & EDA Lead | Dataset inventory, class distribution and split count |
| Likith Kumar Devulapalli | Architecture / Xception Lead | Image-size analysis, channel analysis and sample visualisation |
| Raghava Krishna Battu | Model Comparison Lead | Corrupted image check and duplicate hash analysis |
| Navatej Patel Cheerneni | Evaluation & XAI Lead | Average images, pixel intensity analysis and EDA interpretation |
| Tharun Chinnachinnaiahgari | GitHub / Deployment Lead | Figure saving, weekly summary and GitHub documentation |

## Evidence Files
- `notebooks/Week_02_EDA_and_Data_Quality.ipynb`
- `src/data_preprocessing.py`
- `reports/model_results/week2_class_distribution.csv`
- `reports/model_results/week2_image_inventory_with_metadata.csv`
- `reports/model_results/week2_duplicate_images.csv`
- `reports/model_results/week2_corrupted_images.csv`
- `reports/figures/week2_class_distribution.png`
- `reports/figures/week2_random_train_samples.png`
- `reports/figures/week2_image_dimension_distribution.png`
- `reports/figures/week2_average_images_by_class.png`
- `reports/figures/week2_pixel_intensity_distribution.png`

## Week 2 Outcome
The team completed dataset understanding and prepared evidence for model development. Week 3 will focus on preprocessing, augmentation, class weights and baseline CNN training.
