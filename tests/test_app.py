import io
import json
import unittest
from unittest.mock import patch
from app import app, compare_texts


class DocVerifyTests(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def test_health(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json["ok"])

    def test_missing_extract_file(self):
        response = self.client.post("/api/extract")
        self.assertEqual(response.status_code, 400)
        self.assertFalse(response.json["ok"])

    def test_empty_extract_file(self):
        response = self.client.post("/api/extract", data={"file": (io.BytesIO(b""), "empty.txt")})
        self.assertEqual(response.status_code, 400)

    def test_unsupported_file(self):
        response = self.client.post("/api/extract", data={"file": (io.BytesIO(b"hello"), "payload.exe")})
        self.assertEqual(response.status_code, 400)

    def test_text_compare_missing(self):
        response = self.client.post("/api/plagiarism/text", json={"text1": "", "text2": "hello world"})
        self.assertEqual(response.status_code, 400)

    def test_text_compare_identical(self):
        text = "This is a shared sentence with enough words for matching."
        response = self.client.post("/api/plagiarism/text", json={"text1": text, "text2": text})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json["similarity_percent"], 100.0)

    def test_image_to_pdf_no_files(self):
        response = self.client.post("/api/convert/images-to-pdf")
        self.assertEqual(response.status_code, 400)

    def test_api_not_found(self):
        response = self.client.get("/api/no-such-route")
        self.assertEqual(response.status_code, 404)
        self.assertFalse(response.json["ok"])

    def test_ai_detector_rejects_short_text(self):
        response = self.client.post("/api/ai-detect", json={"text": "Too short."})
        self.assertEqual(response.status_code, 400)
        self.assertFalse(response.json["ok"])

    def test_ai_detector_missing_artifact_is_honest(self):
        with patch("app.load_detector_bundle", return_value=None):
            response = self.client.post("/api/ai-detect", json={"text": "This is a long enough example sentence. " * 8})
        self.assertEqual(response.status_code, 503)
        self.assertIn("model file is missing", response.json["error"])

    def test_comparison_empty_text(self):
        result = compare_texts("", "some words here")
        self.assertEqual(result["similarity_percent"], 0.0)


if __name__ == "__main__":
    unittest.main()
