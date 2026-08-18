# Casting Defect Inspector — Flask Application

A deployment-ready Flask and TensorFlow application for binary visual inspection of
submersible-pump impeller castings.

## Features

- Browser image upload
- Defective and OK probabilities
- Validation-selected decision threshold
- Borderline prediction warning
- Secure in-memory processing
- Optional Grad-CAM explanation
- JSON prediction API
- Health endpoint
- Gunicorn and Docker deployment files

## Required trained files

Place these files in the project root:

```text
casting_defect_model.keras
model_metadata.json
```

The ZIP includes a placeholder `model_metadata.json`. Replace it with the metadata exported
by the corrected training notebook.

## Run locally on Windows

Easiest method: double-click `setup_windows.bat` once, then double-click `run_windows.bat`.

Manual method:

```powershell
cd casting_defect_flask_app
py -3.11 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python app.py
```

Open `http://127.0.0.1:5000`.

## Run locally on macOS or Linux

```bash
cd casting_defect_flask_app
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python app.py
```

## Production command

```bash
gunicorn --config gunicorn.conf.py wsgi:app
```

## API example

```bash
curl -X POST \
  -F "image=@sample_impeller.jpg" \
  http://127.0.0.1:5000/api/predict
```

The response has this structure:

```json
{
  "success": true,
  "decision": "DEFECTIVE",
  "probability_defective": 0.982,
  "probability_ok": 0.018,
  "confidence": 0.982,
  "threshold": 0.61,
  "borderline": false,
  "model_name": "EfficientNetB0"
}
```

## Environment variables

Copy `.env.example` values into your hosting provider's environment configuration.
The Flask app itself does not automatically read `.env`; export the values in your shell or
configure them on the host.

`ENABLE_GRADCAM=false` is recommended for low-memory hosting. Set it to `true` only after the
basic prediction workflow is stable.

## Important limitation

The system is designed for top-view impeller images captured under conditions similar to the
training dataset. It is a research prototype and does not replace a validated industrial
quality-management process.
