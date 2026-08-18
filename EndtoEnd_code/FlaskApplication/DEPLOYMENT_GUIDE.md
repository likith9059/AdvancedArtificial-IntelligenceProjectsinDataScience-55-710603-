# Flask Deployment Guide

## 1. Export the model

Run the corrected training notebook through its export section. Download:

```text
casting_defect_model.keras
model_metadata.json
```

## 2. Place both files in this folder

The final structure must be:

```text
casting_defect_flask_app/
├── app.py
├── model_service.py
├── casting_defect_model.keras
├── model_metadata.json
├── templates/
├── static/
├── requirements.txt
└── ...
```

## 3. Test locally first

Use Python 3.11. Create a virtual environment, install `requirements.txt`, and run `python app.py`.
Test one defective and one OK image before deployment.

## 4. Production configuration

Use this start command:

```text
gunicorn --config gunicorn.conf.py wsgi:app
```

Keep one worker because every Gunicorn worker loads a separate TensorFlow model into memory.
The included configuration uses one worker and two threads.

## 5. Low-memory hosting settings

Start with:

```text
ENABLE_GRADCAM=false
MAX_UPLOAD_MB=8
```

Upload only the application and trained model. Do not upload the dataset, notebook, plots, or
training checkpoints.

## 6. Verify after deployment

Open:

```text
/health
```

Then upload an image through the home page and test the API endpoint `/api/predict`.

## Troubleshooting

### Missing model

Confirm the exact Linux-sensitive filename:

```text
casting_defect_model.keras
```

### Out of memory

Use EfficientNetB0 or MobileNetV2, keep one Gunicorn worker, leave Grad-CAM disabled, and avoid
loading any dataset files in the application.

### Model deserialization error

Use Python 3.11 and `tensorflow-cpu==2.17.1`, matching the training environment.

### Slow first prediction

The model is loaded lazily. The first prediction includes model-loading time; later predictions
reuse the same cached in-process model.
