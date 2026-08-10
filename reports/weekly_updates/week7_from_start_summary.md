# Week 7 Summary: Fresh Training, Grad-CAM and Responsible AI

## Week 7 Aim
The aim of Week 7 was to train the model again from the beginning and then complete explainability and responsible AI analysis. This version does not depend on the Week 6 saved fine-tuned model.

## Work Completed
- Started again from the original casting dataset.
- Created a clean image manifest.
- Created train, validation and test splits.
- Built a new Xception transfer-learning model.
- Trained the classification head.
- Fine-tuned the final Xception backbone layers.
- Selected the decision threshold using validation data.
- Evaluated the fresh model on the held-out test set.
- Saved final metrics, classification report and confusion matrix.
- Completed misclassification review.
- Generated Grad-CAM heatmap overlays.
- Created responsible AI evidence.

## Member Contributions

| Name | Role | Week 7 Contribution |
|---|---|---|
| Sandeep Bommagoni | Data & EDA Lead | Dataset source check, clean manifest and split |
| Likith Kumar Devulapalli | Architecture / Xception Lead | Fresh Xception architecture and Grad-CAM utilities |
| Raghava Krishna Battu | Model Comparison Lead | Head training, fine-tuning and Grad-CAM output generation |
| Navatej Patel Cheerneni | Evaluation & XAI Lead | Threshold selection, final evaluation, error analysis and responsible AI review |
| Tharun Chinnachinnaiahgari | GitHub / Deployment Lead | Documentation, tracker, meeting notes and GitHub evidence |

## Week 7 Outcome
Week 7 now shows a full independent workflow from dataset to trained model to explainability. This is stronger for reproducibility because the Grad-CAM analysis is generated from a model trained again inside the Week 7 notebook.
