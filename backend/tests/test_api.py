import io

from fastapi.testclient import TestClient
from PIL import Image

import api
from tools.types import Box, Detection, GTObject


class FakeDataset:
    ids = ["000001", "000002"]

    def load_image(self, image_id):
        return Image.new("RGB", (400, 200))

    def gts(self, image_id):
        return [GTObject("Car", Box(10, 10, 110, 60)), GTObject("DontCare", Box(0, 0, 5, 5))]


class FakeProposer:
    def propose(self, image, phrases=None, region=None):
        return [Detection("car", Box(10, 10, 110, 60), 0.95, "dino"), Detection("car", Box(150, 10, 250, 60), 0.5, "dino")]


class FakeVerifier:
    def verify(self, image, det):
        return 1.0 if det.score > 0.9 else 0.0


def client():
    api._cache.clear()
    api._cache["dataset"] = FakeDataset()
    api._cache["models"] = (FakeProposer(), FakeVerifier())
    return TestClient(api.app)


def test_images_and_image_endpoints():
    c = client()
    assert c.get("/images").json() == {"total": 2, "ids": ["000001", "000002"]}
    assert c.get("/image/000001").headers["content-type"] == "image/jpeg"
    assert c.get("/image/999999").status_code == 404


def test_label_kitti_returns_three_groups_trace_and_truth():
    out = client().post("/label/000001").json()
    assert len(out["accepted"]) == 1 and len(out["for_human"]) == 1 and out["dropped"] == []
    assert out["trace"][-1].startswith("decision") and out["width"] == 400
    assert [t["class"] for t in out["truth"]] == ["Car"]          # DontCare is left out


def test_label_upload_and_bad_input():
    c = client()
    buffer = io.BytesIO()
    Image.new("RGB", (300, 100)).save(buffer, format="PNG")
    out = c.post("/label-upload", files={"file": ("street.png", buffer.getvalue(), "image/png")}).json()
    assert out["image"] == "street" and out["width"] == 300
    assert c.post("/label-upload", files={"file": ("x.txt", b"hello", "text/plain")}).status_code == 400
    assert c.post("/label/000001?accept_above=0.1&drop_below=0.5").status_code == 400