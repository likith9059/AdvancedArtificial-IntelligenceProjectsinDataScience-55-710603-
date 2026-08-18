from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, render_template, request
from werkzeug.exceptions import RequestEntityTooLarge

from model_service import (
    CastingDefectModel,
    InvalidImageError,
    ModelUnavailableError,
)

ROOT = Path(__file__).resolve().parent
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "bmp", "webp"}

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
LOGGER = logging.getLogger("casting-defect-flask")


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _allowed_filename(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def create_app(test_config: dict[str, Any] | None = None) -> Flask:
    app = Flask(__name__)
    app.config.from_mapping(
        SECRET_KEY=os.getenv("SECRET_KEY", "development-only-change-me"),
        MAX_CONTENT_LENGTH=int(os.getenv("MAX_UPLOAD_MB", "8")) * 1024 * 1024,
        MODEL_PATH=os.getenv(
            "MODEL_PATH",
            str(ROOT / "casting_defect_model.keras"),
        ),
        METADATA_PATH=os.getenv(
            "METADATA_PATH",
            str(ROOT / "model_metadata.json"),
        ),
        ENABLE_GRADCAM=_env_bool("ENABLE_GRADCAM", False),
        BORDERLINE_MARGIN=float(os.getenv("BORDERLINE_MARGIN", "0.10")),
    )

    if test_config:
        app.config.update(test_config)

    model_service = CastingDefectModel(
        Path(app.config["MODEL_PATH"]),
        Path(app.config["METADATA_PATH"]),
        enable_gradcam=bool(app.config["ENABLE_GRADCAM"]),
        borderline_margin=float(app.config["BORDERLINE_MARGIN"]),
    )
    app.extensions["casting_model"] = model_service

    @app.get("/")
    def index():
        return render_template(
            "index.html",
            result=None,
            preview_data_uri=None,
            error=None,
            model_status=model_service.status(),
            metadata=model_service.metadata,
            max_upload_mb=app.config["MAX_CONTENT_LENGTH"] // (1024 * 1024),
        )

    @app.post("/predict")
    def predict_page():
        error: str | None = None
        result = None
        preview_data_uri = None

        uploaded = request.files.get("image")
        if uploaded is None or not uploaded.filename:
            error = "Choose an impeller image before starting the inspection."
        elif not _allowed_filename(uploaded.filename):
            error = "Unsupported file type. Use JPG, JPEG, PNG, BMP, or WebP."
        else:
            try:
                image = model_service.open_image(uploaded.read())
                preview_data_uri = model_service.image_to_data_uri(image)
                result = model_service.predict(image)
            except InvalidImageError as exc:
                error = str(exc)
            except ModelUnavailableError as exc:
                error = str(exc)
                LOGGER.exception("Model unavailable during browser prediction")
            except Exception:
                error = "Prediction failed unexpectedly. Review the server log for details."
                LOGGER.exception("Unexpected browser prediction failure")

        return render_template(
            "index.html",
            result=result,
            preview_data_uri=preview_data_uri,
            error=error,
            model_status=model_service.status(),
            metadata=model_service.metadata,
            max_upload_mb=app.config["MAX_CONTENT_LENGTH"] // (1024 * 1024),
        )

    @app.post("/api/predict")
    def predict_api():
        uploaded = request.files.get("image")
        if uploaded is None or not uploaded.filename:
            return jsonify({"error": "Multipart form field 'image' is required."}), 400
        if not _allowed_filename(uploaded.filename):
            return jsonify({"error": "Unsupported image extension."}), 400

        try:
            image = model_service.open_image(uploaded.read())
            result = model_service.predict(image)
            payload = result.to_dict()
            payload["success"] = True
            return jsonify(payload)
        except InvalidImageError as exc:
            return jsonify({"success": False, "error": str(exc)}), 400
        except ModelUnavailableError as exc:
            return jsonify({"success": False, "error": str(exc)}), 503
        except Exception:
            LOGGER.exception("Unexpected API prediction failure")
            return jsonify(
                {
                    "success": False,
                    "error": "Prediction failed unexpectedly.",
                }
            ), 500

    @app.get("/health")
    def health():
        status = model_service.status()
        state = "ready" if status["loaded"] else (
            "model-present" if status["model_file_exists"] else "model-missing"
        )
        return jsonify({"service": "casting-defect-inspector", "state": state, **status})

    @app.errorhandler(RequestEntityTooLarge)
    def handle_large_upload(_error):
        message = (
            f"The uploaded file exceeds the {app.config['MAX_CONTENT_LENGTH'] // (1024 * 1024)} MB limit."
        )
        if request.path.startswith("/api/"):
            return jsonify({"success": False, "error": message}), 413
        return (
            render_template(
                "index.html",
                result=None,
                preview_data_uri=None,
                error=message,
                model_status=model_service.status(),
                metadata=model_service.metadata,
                max_upload_mb=app.config["MAX_CONTENT_LENGTH"] // (1024 * 1024),
            ),
            413,
        )

    @app.errorhandler(404)
    def not_found(_error):
        if request.path.startswith("/api/"):
            return jsonify({"success": False, "error": "Endpoint not found."}), 404
        return render_template("error.html", code=404, message="Page not found."), 404

    @app.errorhandler(500)
    def internal_error(_error):
        if request.path.startswith("/api/"):
            return jsonify({"success": False, "error": "Internal server error."}), 500
        return render_template("error.html", code=500, message="Internal server error."), 500

    return app


app = create_app()


if __name__ == "__main__":
    app.run(
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "5000")),
        debug=_env_bool("FLASK_DEBUG", False),
    )
