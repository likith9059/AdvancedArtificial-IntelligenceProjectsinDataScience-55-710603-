from __future__ import annotations

import base64
import io
import json
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

try:
    import tensorflow as tf
    from tensorflow import keras
except Exception as exc:  # pragma: no cover - surfaced in the UI at runtime
    tf = None
    keras = None
    TENSORFLOW_IMPORT_ERROR: Exception | None = exc
else:
    TENSORFLOW_IMPORT_ERROR = None


DEFAULT_METADATA: dict[str, Any] = {
    "model_name": "Casting defect classifier",
    "image_size": [224, 224],
    "decision_threshold": 0.5,
    "probability_semantics": "probability_defective",
    "class_mapping": {"0": "ok_front", "1": "def_front"},
}


class ModelUnavailableError(RuntimeError):
    """Raised when the trained model cannot be loaded."""


class InvalidImageError(ValueError):
    """Raised when an uploaded file is not a valid supported image."""


@dataclass(frozen=True)
class PredictionResult:
    decision: str
    probability_defective: float
    probability_ok: float
    confidence: float
    threshold: float
    borderline: bool
    model_name: str
    image_width: int
    image_height: int
    gradcam_data_uri: str | None = None
    gradcam_message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision": self.decision,
            "probability_defective": self.probability_defective,
            "probability_ok": self.probability_ok,
            "confidence": self.confidence,
            "threshold": self.threshold,
            "borderline": self.borderline,
            "model_name": self.model_name,
            "image_width": self.image_width,
            "image_height": self.image_height,
            "gradcam_available": self.gradcam_data_uri is not None,
            "gradcam_message": self.gradcam_message,
        }


class CastingDefectModel:
    """Thread-safe lazy loader and inference service for the Keras classifier."""

    def __init__(
        self,
        model_path: Path,
        metadata_path: Path,
        *,
        enable_gradcam: bool = False,
        borderline_margin: float = 0.10,
    ) -> None:
        self.model_path = model_path
        self.metadata_path = metadata_path
        self.enable_gradcam = enable_gradcam
        self.borderline_margin = max(0.0, float(borderline_margin))

        self._model = None
        self._grad_model = None
        self._metadata: dict[str, Any] = dict(DEFAULT_METADATA)
        self._load_error: str | None = None
        self._loaded = False
        self._lock = threading.RLock()

        self._read_metadata()

    @property
    def metadata(self) -> dict[str, Any]:
        return dict(self._metadata)

    @property
    def model_name(self) -> str:
        return str(self._metadata.get("model_name", DEFAULT_METADATA["model_name"]))

    @property
    def threshold(self) -> float:
        value = float(self._metadata.get("decision_threshold", 0.5))
        return float(np.clip(value, 0.0, 1.0))

    @property
    def image_size(self) -> tuple[int, int]:
        raw = self._metadata.get("image_size", [224, 224])
        if not isinstance(raw, (list, tuple)) or len(raw) != 2:
            return (224, 224)
        height, width = int(raw[0]), int(raw[1])
        if height <= 0 or width <= 0:
            return (224, 224)
        return height, width

    @property
    def load_error(self) -> str | None:
        return self._load_error

    @property
    def is_loaded(self) -> bool:
        return self._loaded and self._model is not None

    @property
    def model_file_exists(self) -> bool:
        return self.model_path.is_file()

    def status(self) -> dict[str, Any]:
        return {
            "model_file_exists": self.model_file_exists,
            "loaded": self.is_loaded,
            "model_name": self.model_name,
            "image_size": list(self.image_size),
            "decision_threshold": self.threshold,
            "gradcam_enabled": self.enable_gradcam,
            "error": self.load_error,
        }

    def _read_metadata(self) -> None:
        if not self.metadata_path.is_file():
            return

        try:
            with self.metadata_path.open("r", encoding="utf-8") as handle:
                user_metadata = json.load(handle)
            if isinstance(user_metadata, dict):
                self._metadata.update(user_metadata)
        except (OSError, json.JSONDecodeError) as exc:
            self._load_error = f"Unable to read model_metadata.json: {exc}"

    def load(self) -> None:
        if self.is_loaded:
            return

        with self._lock:
            if self.is_loaded:
                return

            if TENSORFLOW_IMPORT_ERROR is not None or keras is None:
                self._load_error = (
                    "TensorFlow could not be imported. Confirm that tensorflow-cpu is "
                    f"installed correctly. Original error: {TENSORFLOW_IMPORT_ERROR}"
                )
                raise ModelUnavailableError(self._load_error)

            if not self.model_path.is_file():
                self._load_error = (
                    f"Missing trained model: {self.model_path.name}. Copy the exported "
                    "casting_defect_model.keras file into the Flask project root."
                )
                raise ModelUnavailableError(self._load_error)

            try:
                self._model = keras.models.load_model(self.model_path, compile=False)
                self._loaded = True
                self._load_error = None

                if self.enable_gradcam:
                    try:
                        self._grad_model = self._build_grad_model(self._model)
                    except Exception as exc:  # prediction remains available
                        self._grad_model = None
                        self._load_error = f"Model loaded; Grad-CAM unavailable: {exc}"
            except Exception as exc:
                self._model = None
                self._grad_model = None
                self._loaded = False
                self._load_error = f"Unable to load {self.model_path.name}: {exc}"
                raise ModelUnavailableError(self._load_error) from exc

    @staticmethod
    def open_image(file_bytes: bytes) -> Image.Image:
        if not file_bytes:
            raise InvalidImageError("The uploaded file is empty.")

        try:
            with Image.open(io.BytesIO(file_bytes)) as source:
                source.verify()
            with Image.open(io.BytesIO(file_bytes)) as source:
                image = ImageOps.exif_transpose(source).convert("RGB")
                return image.copy()
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            raise InvalidImageError(
                "The selected file is not a readable JPG, JPEG, PNG, BMP, or WebP image."
            ) from exc

    @staticmethod
    def image_to_data_uri(image: Image.Image, *, quality: int = 90) -> str:
        buffer = io.BytesIO()
        image.convert("RGB").save(buffer, format="JPEG", quality=quality, optimize=True)
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        return f"data:image/jpeg;base64,{encoded}"

    def _prepare_batch(self, image: Image.Image) -> np.ndarray:
        height, width = self.image_size
        resized = image.convert("RGB").resize(
            (width, height),
            Image.Resampling.BILINEAR,
        )
        array = np.asarray(resized, dtype=np.float32)
        return np.expand_dims(array, axis=0)

    def predict(self, image: Image.Image) -> PredictionResult:
        self.load()
        if self._model is None:
            raise ModelUnavailableError(self._load_error or "The model is unavailable.")

        batch = self._prepare_batch(image)

        with self._lock:
            raw_prediction = self._model.predict(batch, verbose=0)

        values = np.asarray(raw_prediction, dtype=np.float32).reshape(-1)
        if values.size != 1:
            raise RuntimeError(
                "The deployed model must return one sigmoid probability per image. "
                f"Received output shape {np.asarray(raw_prediction).shape}."
            )

        probability_defective = float(np.clip(values[0], 0.0, 1.0))
        probability_ok = 1.0 - probability_defective
        threshold = self.threshold
        predicted_defective = probability_defective >= threshold
        decision = "DEFECTIVE" if predicted_defective else "OK"
        confidence = probability_defective if predicted_defective else probability_ok
        borderline = abs(probability_defective - threshold) < self.borderline_margin

        gradcam_data_uri = None
        gradcam_message = None
        if self.enable_gradcam:
            if self._grad_model is None:
                gradcam_message = self._load_error or "Grad-CAM was not initialized."
            else:
                try:
                    heatmap = self._gradcam(batch, predicted_defective)
                    overlay = self._overlay_heatmap(image, heatmap)
                    gradcam_data_uri = self.image_to_data_uri(overlay)
                except Exception as exc:  # explanation must never block prediction
                    gradcam_message = f"Grad-CAM could not be generated: {exc}"

        return PredictionResult(
            decision=decision,
            probability_defective=probability_defective,
            probability_ok=probability_ok,
            confidence=confidence,
            threshold=threshold,
            borderline=borderline,
            model_name=self.model_name,
            image_width=image.width,
            image_height=image.height,
            gradcam_data_uri=gradcam_data_uri,
            gradcam_message=gradcam_message,
        )

    @staticmethod
    def _find_backbone(model):
        candidates = [
            layer
            for layer in model.layers
            if isinstance(layer, keras.Model) and len(getattr(layer, "layers", [])) > 20
        ]
        return candidates[0] if candidates else None

    @staticmethod
    def _last_conv_layer(model):
        for layer in reversed(model.layers):
            try:
                if len(layer.output.shape) == 4:
                    return layer
            except Exception:
                continue
        raise ValueError("No four-dimensional convolutional feature layer was found.")

    def _build_grad_model(self, model):
        backbone = self._find_backbone(model)

        if backbone is None:
            conv_layer = self._last_conv_layer(model)
            return keras.Model(model.inputs, [conv_layer.output, model.output])

        backbone_index = model.layers.index(backbone)
        x = model.input

        for layer in model.layers[1:backbone_index]:
            x = layer(x, training=False)

        conv_layer = self._last_conv_layer(backbone)
        backbone_probe = keras.Model(
            backbone.input,
            [conv_layer.output, backbone.output],
            name="backbone_gradcam_probe",
        )
        conv_output, x = backbone_probe(x)

        for layer in model.layers[backbone_index + 1 :]:
            x = layer(x, training=False)

        return keras.Model(model.input, [conv_output, x], name="casting_gradcam_model")

    def _gradcam(self, batch: np.ndarray, predicted_defective: bool) -> np.ndarray:
        if tf is None or self._grad_model is None:
            raise RuntimeError("Grad-CAM is unavailable.")

        image_tensor = tf.convert_to_tensor(batch)
        with tf.GradientTape() as tape:
            conv_output, prediction = self._grad_model(image_tensor, training=False)
            probability_defective = prediction[:, 0]
            target = (
                probability_defective
                if predicted_defective
                else (1.0 - probability_defective)
            )

        gradients = tape.gradient(target, conv_output)
        if gradients is None:
            raise RuntimeError("TensorFlow did not return gradients for the selected layer.")

        weights = tf.reduce_mean(gradients, axis=(1, 2), keepdims=True)
        heatmap = tf.reduce_sum(weights * conv_output, axis=-1)[0]
        heatmap = tf.maximum(heatmap, 0)
        denominator = tf.reduce_max(heatmap)
        heatmap = tf.where(denominator > 0, heatmap / denominator, heatmap)
        return heatmap.numpy()

    @staticmethod
    def _overlay_heatmap(original: Image.Image, heatmap: np.ndarray) -> Image.Image:
        heat = Image.fromarray(np.uint8(np.clip(heatmap, 0, 1) * 255), mode="L")
        heat = heat.resize(original.size, Image.Resampling.BILINEAR)
        values = np.asarray(heat, dtype=np.float32) / 255.0

        # Lightweight red-yellow heat map without OpenCV or Matplotlib.
        red = np.clip(values * 2.0, 0.0, 1.0)
        green = np.clip((values - 0.25) * 2.0, 0.0, 1.0)
        blue = np.clip((values - 0.75) * 4.0, 0.0, 1.0) * 0.2
        color_map = np.stack([red, green, blue], axis=-1)

        base = np.asarray(original.convert("RGB"), dtype=np.float32) / 255.0
        blended = np.clip(base * 0.58 + color_map * 0.42, 0.0, 1.0)
        return Image.fromarray(np.uint8(blended * 255), mode="RGB")
