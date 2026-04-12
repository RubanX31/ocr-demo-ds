"""
Gemini API wrapper for Discharge Summary POC.

Provides a single reusable :class:`GeminiClient` used by all pipeline stages:
  - Vision LLM text extraction (PDF → text)
  - Reasoning LLM ontology extraction (text → JSON)
  - Hallucination check (text + JSON → validation)

Uses the ``google-generativeai`` library (legacy SDK). All calls are logged
via the Python :mod:`logging` module.
"""

import json
import logging
import re
import time
from pathlib import Path

import google.generativeai as genai

from src.config import GEMINI_API_KEY, GEMINI_MODEL

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Rate-limit retry settings
# ---------------------------------------------------------------------------
_RATE_LIMIT_STATUS_CODE = 429
_RETRY_WAIT_SECONDS = 10
_MAX_RETRIES = 1  # retry once after a 429


class GeminiClient:
    """
    Reusable wrapper around the Google Generative AI SDK.

    Instantiated once at module level as :data:`gemini` and imported by other
    modules (``from src.gemini_client import gemini``).
    """

    def __init__(self):
        """
        Configure the Gemini SDK and set up the default model.

        Reads ``GEMINI_API_KEY`` and ``GEMINI_MODEL`` from :mod:`src.config`.
        Uses a low temperature (0.1) for deterministic, consistent extractions.
        """
        genai.configure(api_key=GEMINI_API_KEY)

        self._model_name = GEMINI_MODEL
        self._generation_config = genai.types.GenerationConfig(
            temperature=0.1,
            max_output_tokens=8192,
        )
        self._model = genai.GenerativeModel(
            model_name=self._model_name,
            generation_config=self._generation_config,
        )
        logger.info(
            "GeminiClient initialised — model=%s, temperature=0.1, "
            "max_output_tokens=8192",
            self._model_name,
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
        Upload a PDF via the Gemini File API and send it with a text prompt.

        The uploaded file is **always** deleted from Google servers after the
        response is received (or on error), to avoid leaking PHI.

        Args:
            prompt:         The text prompt to send alongside the PDF.
            pdf_path:       Absolute path to the local PDF file.
            prompt_version: Version tag for audit logging (default ``"V1"``).

        Returns:
            dict with keys: ``success``, ``response_text``, ``token_count``,
            ``prompt_version``, ``model_used``, ``error``, ``time_ms``.
        """
        result = self._empty_result(prompt_version)
        uploaded_file = None

        try:
            start = time.perf_counter()

            # Upload PDF to Gemini File API
            uploaded_file = genai.upload_file(
                path=str(pdf_path),
                display_name=pdf_path.name,
            )
            logger.info("Uploaded PDF to Gemini File API: %s", pdf_path.name)

            # Send prompt + PDF with retry logic
            response = self._generate_with_retry(
                contents=[uploaded_file, prompt],
            )

            elapsed_ms = int((time.perf_counter() - start) * 1000)

            result["success"] = True
            result["response_text"] = response.text
            result["token_count"] = self._extract_token_count(response)
            result["time_ms"] = elapsed_ms

            logger.info(
                "PDF prompt completed — model=%s, version=%s, "
                "tokens=%d, time=%dms",
                self._model_name,
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
                self._model_name,
                prompt_version,
                elapsed_ms,
                exc,
            )

        finally:
            # Always clean up the uploaded file
            if uploaded_file is not None:
                try:
                    genai.delete_file(uploaded_file.name)
                    logger.info(
                        "Deleted uploaded file from Gemini: %s",
                        uploaded_file.name,
                    )
                except Exception as cleanup_exc:
                    logger.warning(
                        "Failed to delete uploaded file: %s", cleanup_exc
                    )

        return result

    def send_prompt_with_text(
        self,
        prompt: str,
        prompt_version: str = "V1",
    ) -> dict:
        """
        Send a text-only prompt to Gemini (no file upload).

        Used by the Reasoning LLM (ontology extraction) and the Hallucination
        Check stages.

        Args:
            prompt:         The full text prompt to send.
            prompt_version: Version tag for audit logging (default ``"V1"``).

        Returns:
            dict with keys: ``success``, ``response_text``, ``token_count``,
            ``prompt_version``, ``model_used``, ``error``, ``time_ms``.
        """
        result = self._empty_result(prompt_version)

        try:
            start = time.perf_counter()

            response = self._generate_with_retry(contents=[prompt])

            elapsed_ms = int((time.perf_counter() - start) * 1000)

            result["success"] = True
            result["response_text"] = response.text
            result["token_count"] = self._extract_token_count(response)
            result["time_ms"] = elapsed_ms

            logger.info(
                "Text prompt completed — model=%s, version=%s, "
                "tokens=%d, time=%dms",
                self._model_name,
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
                self._model_name,
                prompt_version,
                elapsed_ms,
                exc,
            )

        return result

    def parse_json_response(self, response_text: str) -> dict:
        """
        Extract and parse JSON from a Gemini response string.

        Handles the common case where Gemini wraps JSON output inside
        markdown code fences (``\\`\\`\\`json ... \\`\\`\\```).

        Args:
            response_text: Raw text from the Gemini response.

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

    def _empty_result(self, prompt_version: str) -> dict:
        """Return a blank result dict with all expected keys."""
        return {
            "success": False,
            "response_text": None,
            "token_count": 0,
            "prompt_version": prompt_version,
            "model_used": self._model_name,
            "error": None,
            "time_ms": 0,
        }

    def _generate_with_retry(self, contents: list):
        """
        Call ``model.generate_content`` with a single retry on HTTP 429
        (rate limit exceeded).

        Args:
            contents: The content list to pass to ``generate_content``.

        Returns:
            The Gemini response object.

        Raises:
            The original exception if the retry also fails or if the error
            is not a rate-limit error.
        """
        for attempt in range(_MAX_RETRIES + 1):
            try:
                response = self._model.generate_content(contents)
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
        # google-generativeai raises google.api_core.exceptions.ResourceExhausted
        # for 429 errors. Also check the string representation as a fallback.
        exc_type_name = type(exc).__name__
        if exc_type_name in ("ResourceExhausted", "TooManyRequests"):
            return True
        if "429" in str(exc) or "rate limit" in str(exc).lower():
            return True
        return False

    @staticmethod
    def _extract_token_count(response) -> int:
        """
        Safely extract total token count from a Gemini response.

        Tries ``response.usage_metadata.total_token_count`` first, then
        falls back to ``candidates_token_count``, and finally returns 0
        if neither is available.
        """
        try:
            metadata = response.usage_metadata
            if hasattr(metadata, "total_token_count"):
                return metadata.total_token_count or 0
            if hasattr(metadata, "candidates_token_count"):
                return metadata.candidates_token_count or 0
        except (AttributeError, TypeError):
            pass
        return 0


# ============================================================================
# Module-level convenience instance — import as:
#     from src.gemini_client import gemini
# ============================================================================
gemini = GeminiClient()
