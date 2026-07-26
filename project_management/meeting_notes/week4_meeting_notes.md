# Week 4 Meeting Notes

## Meeting Topic
Xception Transfer Learning

## Attendees
Sandeep Bommagoni, Likith Kumar Devulapalli, Raghava Krishna Battu, Navatej Patel Cheerneni, Tharun Chinnachinnaiahgari

## Completed This Week
1. The Week 3 preprocessing approach was reused for consistency.
2. Image size was changed to 299 × 299 for Xception.
3. A frozen-backbone Xception transfer-learning model was created.
4. A custom binary classification head was added.
5. Class weights were used during training.
6. ModelCheckpoint, EarlyStopping, ReduceLROnPlateau and CSVLogger callbacks were added.
7. Xception validation results were saved.
8. Training curves and confusion matrix were generated.
9. A comparison structure against the Week 3 baseline was prepared.

## Key Discussion
The team agreed that Week 4 should only train the classification head with the Xception backbone frozen. Full fine-tuning will be done later after comparing multiple models. This avoids changing too many variables at once.

## Blockers
ImageNet weights may require internet access in Kaggle. If ImageNet weights cannot be loaded, the notebook falls back to random initialisation, but the team should document this and ideally enable internet to use transfer learning properly.

## Next Week Plan
Week 5 will compare Xception with other transfer-learning models such as MobileNetV2 and EfficientNetB0.
