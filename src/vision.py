# src/vision.py
import logging
import google.generativeai as genai

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class VisionExtractor:
    """
    Uses Gemini Vision to extract text (and questions) from an image.
    """

    def __init__(self, model_name: str = "gemini-2.5-flash"):
        self.model_name = model_name
        try:
            self.model = genai.GenerativeModel(model_name)
        except Exception:
            self.model = None

    def _build_prompt(self) -> str:
        return (
            "You are given an image of a question paper, notes or a textbook page. "
            "Extract ALL readable text from the image. "
            "If there are questions, write each one on its own line starting with 'QUESTION:'. "
            "Do NOT add any new text that is not present in the image."
        )

    def extract_text(self, uploaded_file) -> str:
        uploaded_file.seek(0)
        image_bytes = uploaded_file.read()
        if not image_bytes:
            return ""

        prompt = self._build_prompt()

        try:
            if self.model:
                response = self.model.generate_content([
                    prompt,
                    {
                        "mime_type": uploaded_file.type,
                        "data": image_bytes
                    }
                ])
                text = getattr(response, "text", "") or str(response)
                return text.strip()
            else:
                resp = genai.generate_text(
                    model=self.model_name,
                    prompt=prompt + "\n\n[IMAGE DATA OMITTED]",
                )
                text = getattr(resp, "text", "") or str(resp)
                return text.strip()
        except Exception as e:
            logger.exception("Vision extraction failed: %s", e)
            return ""
