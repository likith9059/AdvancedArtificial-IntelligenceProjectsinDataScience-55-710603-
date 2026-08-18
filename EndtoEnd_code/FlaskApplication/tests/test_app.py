from io import BytesIO

from PIL import Image

from app import create_app


def _image_bytes() -> BytesIO:
    buffer = BytesIO()
    Image.new("RGB", (32, 32), "white").save(buffer, format="PNG")
    buffer.seek(0)
    return buffer


def test_home_page_loads_without_model():
    app = create_app({"TESTING": True, "MODEL_PATH": "/tmp/nonexistent.keras"})
    client = app.test_client()
    response = client.get("/")
    assert response.status_code == 200
    assert b"Casting Defect Inspector" in response.data


def test_api_requires_image():
    app = create_app({"TESTING": True, "MODEL_PATH": "/tmp/nonexistent.keras"})
    client = app.test_client()
    response = client.post("/api/predict", data={})
    assert response.status_code == 400
    assert "error" in response.get_json()


def test_invalid_extension_is_rejected():
    app = create_app({"TESTING": True, "MODEL_PATH": "/tmp/nonexistent.keras"})
    client = app.test_client()
    response = client.post(
        "/api/predict",
        data={"image": (_image_bytes(), "image.txt")},
        content_type="multipart/form-data",
    )
    assert response.status_code == 400
