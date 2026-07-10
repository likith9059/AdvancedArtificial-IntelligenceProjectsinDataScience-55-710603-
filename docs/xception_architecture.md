# Xception Architecture Plan

## Model Flow
```text
Input image 299x299x3
 → Data augmentation
 → Xception base with ImageNet weights
 → Global Average Pooling
 → Dense layer
 → Dropout
 → Sigmoid output
 → Defective / OK prediction
```

## Training Plan
1. Train frozen Xception base with custom head.
2. Fine-tune upper layers using small learning rate.
3. Use EarlyStopping, ReduceLROnPlateau and ModelCheckpoint.
