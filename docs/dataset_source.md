# Dataset Source and Folder Structure

## Expected Dataset
Kaggle real-life industrial dataset of casting product images.

## Expected Kaggle Path
`/kaggle/input/real-life-industrial-dataset-of-casting-product/casting_data/casting_data`

## Expected Folder Structure
```text
casting_data/
├── train/
│   ├── def_front/
│   └── ok_front/
└── test/
    ├── def_front/
    └── ok_front/
```

## Class Mapping
| Folder | Label | Meaning |
|---|---|---|
| def_front | Defective | casting product has visible defect |
| ok_front | OK | casting product appears acceptable |

## Week 2 EDA Plan
Class counts, sample images, image sizes, corrupt file check, duplicate hash check, pixel intensity and average image analysis.
