from unittest.mock import patch, MagicMock

from app.services import gemini


class TestGeminiAvailability:
    def test_not_available_without_api_key(self):
        with patch.object(gemini, "_client", None), \
             patch.object(gemini, "GEMINI_API_KEY", ""):
            assert gemini.is_available() is False


class TestSummarize:
    def test_returns_config_message_when_no_api_key(self):
        with patch.object(gemini, "_client", None), \
             patch.object(gemini, "GEMINI_API_KEY", ""):
            result = gemini.summarize("Title", "Content")
            assert "not configured" in result.lower()

    def test_returns_error_message_on_api_failure(self):
        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = Exception("API error")

        with patch.object(gemini, "_client", mock_client), \
             patch.object(gemini, "GEMINI_API_KEY", "test-key"):
            result = gemini.summarize("Title", "Content")
            assert "unavailable" in result.lower() or "error" in result.lower()

    def test_truncates_long_content(self):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Summary"
        mock_client.models.generate_content.return_value = mock_response

        long_content = "x" * 5000

        with patch.object(gemini, "_client", mock_client), \
             patch.object(gemini, "GEMINI_API_KEY", "test-key"):
            gemini.summarize("Title", long_content)

            call_args = mock_client.models.generate_content.call_args
            contents = call_args[1]["contents"] if "contents" in call_args[1] else call_args[0][1]
            assert len(contents) < 3000
