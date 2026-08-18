# Automated Casting Defect Detection Using Deep Learning and Computer Vision

![Clean System Architecture](architecture/Clean_System_Architecture.png)

**Module Leader:** Royce Copley  
**Level:** 7  
**Module Name:** Advanced Artificial Intelligence Projects in Data Science  
**Module Code:** 55-710603  
**Assignment Title:** AI Project and Pitch Group Work and Individual Report  
**Weighting:** 100%  
**Submission Format:** Word/PDF report, PowerPoint presentation and complete source code  

---

## 1. Project Overview

This repository contains the complete end-to-end source code and Flask application for an AI-assisted casting defect inspection system.

The project classifies casting product images into:

- `def_front` — defective casting product
- `ok_front` — acceptable casting product

The system uses deep learning and computer vision to support manufacturing quality inspection. It compares CNN-based models, selects a fine-tuned Xception model, evaluates performance using classification metrics, and provides a web-based demo where users can upload a casting image and receive an inspection result.

---

## 2. End-to-End Code Location

The complete notebook and Flask application are stored in the following GitHub folder:

```text
EndtoEnd_code/
```

GitHub folder link:

```text
https://github.com/likith9059/AdvancedArtificial-IntelligenceProjectsinDataScience-55-710603-/tree/main/EndtoEnd_code
```

This folder contains the main `.ipynb` notebook and the Flask application files.

---

## 3. Problem We Are Solving

Casting is a manufacturing process where liquid material is poured into a mould and allowed to solidify. During casting, defects such as blow holes, pinholes, burrs, shrinkage defects, mould material defects, pouring metal defects and metallurgical defects can occur.

Manual quality inspection is often:

- time-consuming,
- dependent on human judgement,
- not always fully consistent,
- difficult to scale for large production volumes.

A missed defective product can create rework, rejection of orders and financial loss. This project addresses the problem by building an AI system that screens casting images and supports inspectors by identifying likely defective products.

The system is not intended to replace inspectors. It is designed as a **human-in-the-loop decision-support tool**.

---

## 4. Dataset

The project uses the Kaggle casting product image dataset.

**Dataset:** Casting Product Image Data for Quality Inspection  
**Kaggle Link:** https://www.kaggle.com/datasets/ravirajsinh45/real-life-industrial-dataset-of-casting-product

The images are top-view images of submersible pump impellers.

### Dataset Classes

| Class | Meaning |
|---|---|
| `def_front` | Defective casting product |
| `ok_front` | Acceptable casting product |

### Full Pipeline Dataset Split

| Split | def_front | ok_front | Total |
|---|---:|---:|---:|
| Train | 3,758 | 2,875 | 6,633 |
| Test | 453 | 262 | 715 |
| **Total** | **4,211** | **3,137** | **7,348** |

The dataset also includes a 512×512 unaugmented version with:

- 519 `ok_front` images
- 781 `def_front` images

---

## 5. System Architecture

The clean system architecture is included in this repository:

```text
architecture/Clean_System_Architecture.png
```

GitHub architecture image:

```text
https://github.com/likith9059/AdvancedArtificial-IntelligenceProjectsinDataScience-55-710603-/blob/main/architecture/Clean_System_Architecture.png
```

### Architecture Workflow

```text
Dataset
↓
EDA + Quality Checks
↓
Preprocessing
↓
Train / Validation / Test Split
↓
Baseline CNN, Xception and Comparison Models
↓
Technical Evaluation
↓
Error Analysis
↓
Best Model Selection
↓
Grad-CAM Explainability
↓
Saved Best Model
↓
Flask / Gradio Demo App
↓
Human-in-the-loop Review
```

The architecture also includes a GitHub evidence layer:

```text
commits • issues • project board • pull requests • meeting notes • README
```

---

## 6. Team Roles

| Member | Role | Main Evidence |
|---|---|---|
| Sandeep Bommagoni | Data & EDA Lead | Dataset source, class structure, EDA and quality checks |
| Likith Kumar Devulapalli | Architecture / Xception Lead | System architecture, Xception model design and implementation |
| Raghava Krishna Battu | Model Comparison Lead | Baseline CNN, model comparison and training experiments |
| Navatej Patel Cheerneni | Evaluation, Grad-CAM & Ethics Lead | Final metrics, confusion matrix, Grad-CAM and responsible AI |
| Tharun Chinnachinnaiahgari | GitHub, Deployment & Integration Lead | GitHub evidence, README, Flask app and deployment preparation |

---

## 7. AI Techniques Used

| Technique / Model | Purpose |
|---|---|
| Custom CNN | Baseline model trained from scratch |
| MobileNetV2 | Lightweight transfer-learning comparison model |
| EfficientNetB0 | Efficient transfer-learning comparison model |
| DenseNet121 | Deeper CNN comparison model |
| Xception | Main selected transfer-learning model |
| Fine-tuning | Improves the selected pretrained model for the casting dataset |
| Grad-CAM | Shows visual evidence of what image regions influenced predictions |
| Flask Web App | Provides the live demo interface for image upload and prediction |

---

## 8. Final Selected Model

The final selected model is:

```text
Xception_fine_tuned
```

### Final Test Results

| Metric | Result |
|---|---:|
| Accuracy | 99.86% |
| Precision | 0.9962 |
| Recall | 1.0000 |
| F1-score | 0.9981 |
| AUC | 1.0000 |
| Misclassified Images | 1 out of 715 |

The model achieved strong performance on the test set. However, duplicate image entries were detected during data analysis, so the results are discussed carefully and external validation is recommended before real-world industrial use.

---

## 9. Model Files

Large `.keras` model files should not normally be uploaded directly to GitHub. They should be downloaded from the external OneDrive link and placed inside the `models/` folder.

### Model Download Instruction

Download the trained model file from the OneDrive model link provided by the team, then place it here:

```text
EndtoEnd_code/models/
```

Expected structure:

```text
EndtoEnd_code/models/
└── casting_defect_best_model.keras
```

There is also a placeholder file in the repository:

```text
EndtoEnd_code/MODEL_FILE_GOES_HERE.txt
```

Use that file to record the actual OneDrive model link for tutors and team members.

Example format inside `MODEL_FILE_GOES_HERE.txt`:

```text
Download the trained model from:
PASTE_ONEDRIVE_MODEL_LINK_HERE

After downloading, place the .keras file inside:
EndtoEnd_code/models/
```

---

## 10. Repository / Application Structure

The `EndtoEnd_code/` folder contains the runnable end-to-end implementation.

```text
EndtoEnd_code/
├── app.py
├── model_service.py
├── model_metadata.json
├── requirements.txt
├── requirements-dev.txt
├── setup_windows.bat
├── run_windows.bat
├── run_linux_mac.sh
├── Dockerfile
├── Procfile
├── render.yaml
├── runtime.txt
├── gunicorn.conf.py
├── DEPLOYMENT_GUIDE.md
├── repair_keras_model.py
├── MODEL_FILE_GOES_HERE.txt
├── static/
├── templates/
├── tests/
└── models/
    └── trained model file goes here
```

---

## 11. Setup Instructions

### Recommended Python Version

Use Python 3.11.

```bash
python --version
```

Expected:

```text
Python 3.11.x
```

Python 3.13 may cause TensorFlow installation problems, so Python 3.11 is recommended.

---

### Windows PowerShell Setup

```powershell
cd EndtoEnd_code

python -m venv .venv

.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip

pip install -r requirements.txt
```

---

### Windows CMD Setup

```cmd
cd EndtoEnd_code

python -m venv .venv

.venv\Scripts\activate.bat

python -m pip install --upgrade pip

pip install -r requirements.txt
```

---

### Linux / macOS Setup

```bash
cd EndtoEnd_code

python3 -m venv .venv

source .venv/bin/activate

python -m pip install --upgrade pip

pip install -r requirements.txt
```

---

## 12. How to Run the Flask Application

Before running the app:

1. Download the trained model from the OneDrive model link.
2. Place the `.keras` model inside:

```text
EndtoEnd_code/models/
```

Then run:

```bash
python app.py
```

Open the local URL shown in the terminal, usually:

```text
http://127.0.0.1:5000
```

The app allows users to:

1. upload a casting image,
2. validate the uploaded image,
3. resize and preprocess the image,
4. run model inference,
5. display the predicted class,
6. show defective and OK probabilities,
7. support manual quality review.

---

## 13. Windows Helper Files

The project also includes helper batch files.

### Setup

```cmd
setup_windows.bat
```

### Run

```cmd
run_windows.bat
```

---

## 14. Linux / macOS Helper File

```bash
bash run_linux_mac.sh
```

---

## 15. End-to-End Notebook

The notebook in `EndtoEnd_code/` shows the full modelling workflow.

The notebook covers:

```text
1. Dataset loading
2. Exploratory data analysis
3. Image quality checks
4. Data preprocessing
5. Class weighting
6. Custom CNN baseline
7. Transfer-learning model comparison
8. Xception fine-tuning
9. Final test evaluation
10. Confusion matrix and classification report
11. Error analysis
12. Grad-CAM explainability
13. Model saving
14. Flask deployment preparation
```

---

## 16. Grad-CAM Explainability

Grad-CAM is used to visualise which image regions influenced the CNN prediction.

This supports responsible AI because it helps the team check whether the model focuses on meaningful casting regions rather than irrelevant background areas.

Important note:

```text
Grad-CAM is useful for explanation, but it is not perfect proof of model reasoning.
```

Therefore, Grad-CAM outputs should support human inspection rather than replace expert judgement.

---

## 17. Live Demo Plan

The group presentation includes a live demonstration.

Demo steps:

1. Open the Flask application.
2. Upload a casting product image.
3. Click the inspection button.
4. Show the model decision: `def_front` or `ok_front`.
5. Show defective probability and OK probability.
6. Explain the confidence threshold.
7. Discuss human-in-the-loop quality review.

This satisfies the presentation requirement to run the system live and show new input/output rather than relying only on screenshots.

---

## 18. Assessment Learning Outcome Mapping

| Learning Outcome | How This Project Satisfies It |
|---|---|
| LO1 | Explores CNNs, transfer learning, Xception, model comparison and Grad-CAM |
| LO2 | Implements a practical AI solution using Python, TensorFlow/Keras and Flask |
| LO3 | Evaluates performance using accuracy, precision, recall, F1-score, AUC and confusion matrix |
| LO4 | Discusses false negatives, dataset limitations, explainability, bias and human oversight |

---

## 19. Project Management and GitHub Evidence

GitHub is used to document weekly project progress.

Evidence includes:

- weekly milestones,
- assigned issues,
- closed issues,
- commit history,
- project-management tracker,
- member contribution records,
- meeting notes,
- README updates,
- source code,
- notebook outputs.

This supports the module rule:

> Work that is not documented on GitHub does not exist.

---

## 20. Limitations

The project has the following limitations:

- duplicate image entries were detected during EDA,
- the dataset includes augmented/similar images,
- test performance may be higher than real production performance,
- lighting and camera setup changes may affect predictions,
- the model is trained on a limited casting-image dataset,
- Grad-CAM is supportive but not a complete explanation,
- external validation is required before real industrial deployment.

---

## 21. Future Work

Future work should include:

- testing on new factory images,
- collecting more diverse defect examples,
- validating across different lighting and camera setups,
- adding defect localisation or segmentation,
- improving Grad-CAM display inside the web app,
- adding model monitoring and drift detection,
- creating a human-in-the-loop inspection dashboard,
- preparing a controlled deployment in a production-like environment.

---

## 22. Quick Start

```bash
git clone https://github.com/likith9059/AdvancedArtificial-IntelligenceProjectsinDataScience-55-710603-.git

cd AdvancedArtificial-IntelligenceProjectsinDataScience-55-710603-/EndtoEnd_code

python -m venv .venv

# Windows PowerShell
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Download model from OneDrive and place it in EndtoEnd_code/models/

python app.py
```

Open:

```text
http://127.0.0.1:5000
```

---

## 23. Final Project Statement

This project demonstrates a complete AI-assisted quality inspection workflow for casting defect detection. It includes dataset understanding, preprocessing, CNN modelling, transfer learning, model comparison, Xception fine-tuning, technical evaluation, Grad-CAM explainability, live Flask demo preparation, responsible AI discussion and GitHub-based project management evidence.
