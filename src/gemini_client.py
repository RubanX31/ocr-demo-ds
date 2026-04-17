"""
Groq API wrapper for Discharge Summary POC.

Provides a single reusable :class:`GeminiClient` used by all pipeline stages:
  - Vision LLM text extraction (PDF → image → text)
  - Reasoning LLM ontology extraction (text → JSON)
  - Hallucination check (text + JSON → validation)

Uses the ``groq`` Python SDK. PDF pages are converted to JPEG images via
``pdf2image`` before being sent to the vision model. All calls are logged
via the Python :mod:`logging` module.

.. note::
   The class is still named ``GeminiClient`` and the module-level instance is
   still ``gemini`` so that **no other files in the pipeline need to change**.
"""

import base64
import io
import json
import logging
import re
import time
from pathlib import Path

from groq import Groq
from pdf2image import convert_from_path

from src.config import GROQ_API_KEY, GROQ_MODEL_VISION, GROQ_MODEL_TEXT, POPPLER_DIR

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Rate-limit retry settings
# ---------------------------------------------------------------------------
_RATE_LIMIT_STATUS_CODE = 429
_RETRY_WAIT_SECONDS = 10
_MAX_RETRIES = 1  # retry once after a 429


class GeminiClient:
    """
    Reusable wrapper around the Groq SDK.

    Instantiated once at module level as :data:`gemini` and imported by other
    modules (``from src.gemini_client import gemini``).
    """

    def __init__(self):
        """
        Initialise the Groq client.

        Reads ``GROQ_API_KEY``, ``GROQ_MODEL_VISION``, and
        ``GROQ_MODEL_TEXT`` from :mod:`src.config`.
        """
        self._client = Groq(api_key=GROQ_API_KEY)
        self._vision_model = GROQ_MODEL_VISION
        self._text_model = GROQ_MODEL_TEXT

        logger.info(
            "GeminiClient initialised — vision_model=%s, text_model=%s",
            self._vision_model,
            self._text_model,
        )

    # ================================================================
    # Public Methods
    # ================================================================

    def send_prompt_with_pdf(
        self,
        prompt: str,
        pdf_path: Path,
        prompt_version: str = "V1",
    ) -> dict:
        """
        Convert the first page of a PDF to JPEG and send it with a text
        prompt to the Groq vision model.

        Args:
            prompt:         The text prompt to send alongside the PDF.
            pdf_path:       Absolute path to the local PDF file.
            prompt_version: Version tag for audit logging (default ``"V1"``).

        Returns:
            dict with keys: ``success``, ``response_text``, ``token_count``,
            ``prompt_version``, ``model_used``, ``error``, ``time_ms``.
        """
        result = self._empty_result(prompt_version, self._vision_model)

        try:
            start = time.perf_counter()

            # Convert first page of PDF to JPEG image
            images = convert_from_path(
                str(pdf_path), dpi=150, first_page=1, last_page=1,
                poppler_path=str(POPPLER_DIR),
            )
            first_page = images[0]

            # Encode image to base64 JPEG string
            buffer = io.BytesIO()
            first_page.save(buffer, format="JPEG")
            base64_image = base64.b64encode(buffer.getvalue()).decode("utf-8")

            logger.info(
                "Converted first page of PDF to JPEG for vision model: %s",
                pdf_path.name,
            )

            # Build message with image_url content type
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt,
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}",
                            },
                        },
                    ],
                }
            ]

            # Send to Groq vision model with retry logic
            response = self._chat_with_retry(
                model=self._vision_model,
                messages=messages,
            )

            elapsed_ms = int((time.perf_counter() - start) * 1000)

            result["success"] = True
            result["response_text"] = response.choices[0].message.content
            result["token_count"] = self._extract_token_count(response)
            result["time_ms"] = elapsed_ms

            logger.info(
                "PDF prompt completed — model=%s, version=%s, "
                "tokens=%d, time=%dms",
                self._vision_model,
                prompt_version,
                result["token_count"],
                elapsed_ms,
            )

        except Exception as exc:
            elapsed_ms = int((time.perf_counter() - start) * 1000)
            result["error"] = str(exc)
            result["time_ms"] = elapsed_ms
            logger.error(
                "PDF prompt FAILED — model=%s, version=%s, time=%dms, "
                "error=%s",
                self._vision_model,
                prompt_version,
                elapsed_ms,
                exc,
            )

        return result

    def send_prompt_with_text(
        self,
        prompt: str,
        prompt_version: str = "V1",
    ) -> dict:
        """
        Send a text-only prompt to the Groq text model.

        Used by the Reasoning LLM (ontology extraction) and the Hallucination
        Check stages.

        Args:
            prompt:         The full text prompt to send.
            prompt_version: Version tag for audit logging (default ``"V1"``).

        Returns:
            dict with keys: ``success``, ``response_text``, ``token_count``,
            ``prompt_version``, ``model_used``, ``error``, ``time_ms``.
        """
        result = self._empty_result(prompt_version, self._text_model)

        try:
            start = time.perf_counter()

            messages = [
                {
                    "role": "user",
                    "content": prompt,
                }
            ]

            response = self._chat_with_retry(
                model=self._text_model,
                messages=messages,
            )

            elapsed_ms = int((time.perf_counter() - start) * 1000)

            result["success"] = True
            result["response_text"] = response.choices[0].message.content
            result["token_count"] = self._extract_token_count(response)
            result["time_ms"] = elapsed_ms

            logger.info(
                "Text prompt completed — model=%s, version=%s, "
                "tokens=%d, time=%dms",
                self._text_model,
                prompt_version,
                result["token_count"],
                elapsed_ms,
            )

        except Exception as exc:
            elapsed_ms = int((time.perf_counter() - start) * 1000)
            result["error"] = str(exc)
            result["time_ms"] = elapsed_ms
            logger.error(
                "Text prompt FAILED — model=%s, version=%s, time=%dms, "
                "error=%s",
                self._text_model,
                prompt_version,
                elapsed_ms,
                exc,
            )

        return result

    def parse_json_response(self, response_text: str) -> dict:
        """
        Extract and parse JSON from a model response string.

        Handles the common case where the model wraps JSON output inside
        markdown code fences (``\\`\\`\\`json ... \\`\\`\\```).

        Args:
            response_text: Raw text from the model response.

        Returns:
            Parsed ``dict`` on success, or
            ``{"error": "parse_failed", "raw": response_text}`` on failure.
        """
        if not response_text:
            return {"error": "parse_failed", "raw": response_text}

        cleaned = response_text.strip()

        # Strip markdown code fences if present
        # Handles: ```json\n{...}\n``` or ```\n{...}\n```
        fence_pattern = r"^```(?:json)?\s*\n?(.*?)\n?\s*```$"
        match = re.match(fence_pattern, cleaned, re.DOTALL)
        if match:
            cleaned = match.group(1).strip()

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as exc:
            logger.warning("JSON parse failed: %s", exc)
            return {"error": "parse_failed", "raw": response_text}

    # ================================================================
    # Private Helpers
    # ================================================================

    def _empty_result(self, prompt_version: str, model_name: str) -> dict:
        """Return a blank result dict with all expected keys."""
        return {
            "success": False,
            "response_text": None,
            "token_count": 0,
            "prompt_version": prompt_version,
            "model_used": model_name,
            "error": None,
            "time_ms": 0,
        }

    def _chat_with_retry(self, model: str, messages: list):
        """
        Call ``client.chat.completions.create`` with a single retry on
        HTTP 429 (rate limit exceeded).

        Args:
            model:    The Groq model name to use.
            messages: The messages list for the chat completion.

        Returns:
            The Groq ChatCompletion response object.

        Raises:
            The original exception if the retry also fails or if the error
            is not a rate-limit error.
        """
        for attempt in range(_MAX_RETRIES + 1):
            try:
                response = self._client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=0.1,
                    max_tokens=8192,
                )
                return response
            except Exception as exc:
                if self._is_rate_limit_error(exc) and attempt < _MAX_RETRIES:
                    logger.warning(
                        "Rate limited (429). Waiting %ds before retry "
                        "(attempt %d/%d)...",
                        _RETRY_WAIT_SECONDS,
                        attempt + 1,
                        _MAX_RETRIES,
                    )
                    time.sleep(_RETRY_WAIT_SECONDS)
                    continue
                raise

    @staticmethod
    def _is_rate_limit_error(exc: Exception) -> bool:
        """Check if an exception is a 429 rate-limit error."""
        exc_type_name = type(exc).__name__
        if exc_type_name in ("RateLimitError", "TooManyRequests"):
            return True
        if "429" in str(exc) or "rate limit" in str(exc).lower():
            return True
        return False

    @staticmethod
    def _extract_token_count(response) -> int:
        """
        Safely extract total token count from a Groq ChatCompletion response.

        Tries ``response.usage.total_tokens`` first, then falls back to
        ``completion_tokens``, and finally returns 0 if neither is available.
        """
        try:
            usage = response.usage
            if hasattr(usage, "total_tokens") and usage.total_tokens:
                return usage.total_tokens
            if hasattr(usage, "completion_tokens") and usage.completion_tokens:
                return usage.completion_tokens
        except (AttributeError, TypeError):
            pass
        return 0


# ============================================================================
# Module-level convenience instance — import as:
#     from src.gemini_client import gemini
# ============================================================================
gemini = GeminiClient()
