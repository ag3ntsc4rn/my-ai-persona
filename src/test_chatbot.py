import os
import sys
import types
import pytest
from unittest.mock import patch, MagicMock, mock_open


@pytest.fixture(autouse=True)
def patch_env(monkeypatch, tmp_path):
    # Patch environment variables and file paths
    monkeypatch.setenv("PROFILE_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("PROFILE_PDF", "Profile.pdf")
    monkeypatch.setenv("SUMMARY_TXT", "summary.txt")
    monkeypatch.setenv("PROFILE_NAME", "Test Name")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-test")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-test")
    monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")
    monkeypatch.setenv("GOOGLE_API_BASE_URL", "https://fake.url/")
    monkeypatch.setenv("OPENAI_API_KEY", "fake-openai-key")
    # Create dummy files
    (tmp_path / "Profile.pdf").write_text("PDF page 1 text", encoding="utf-8")
    (tmp_path / "summary.txt").write_text("Summary text", encoding="utf-8")


def test_load_pdf_text(monkeypatch, tmp_path):
    import chatbot

    # Patch PdfReader to simulate PDF reading
    class DummyPage:
        def extract_text(self):
            return "page text"

    class DummyReader:
        pages = [DummyPage(), DummyPage()]

    monkeypatch.setattr(chatbot, "PdfReader", lambda path: DummyReader())
    result = chatbot.load_pdf_text("dummy.pdf")
    assert result == "page textpage text"


def test_load_text_file(tmp_path):
    import chatbot

    file_path = tmp_path / "test.txt"
    file_path.write_text("hello world", encoding="utf-8")
    assert chatbot.load_text_file(str(file_path)) == "hello world"


def test_build_system_prompt():
    import chatbot

    prompt = chatbot.build_system_prompt("Alice", "summary", "linkedin")
    assert "Alice" in prompt
    assert "summary" in prompt
    assert "linkedin" in prompt


def test_build_evaluator_system_prompt():
    import chatbot

    prompt = chatbot.build_evaluator_system_prompt("Bob", "sum", "li")
    assert "Bob" in prompt
    assert "sum" in prompt
    assert "li" in prompt


def test_evaluator_user_prompt():
    import chatbot

    result = chatbot.evaluator_user_prompt("reply", "msg", "hist")
    assert "reply" in result
    assert "msg" in result
    assert "hist" in result


def test_get_openai_reply(monkeypatch):
    import chatbot

    fake_openai = MagicMock()
    fake_openai.chat.completions.create.return_value.choices = [
        types.SimpleNamespace(message=types.SimpleNamespace(content="the reply"))
    ]
    monkeypatch.setattr(chatbot, "openai", fake_openai)
    messages = [{"role": "user", "content": "hi"}]
    assert chatbot.get_openai_reply(messages, system="sys") == "the reply"


def test_evaluate(monkeypatch):
    import chatbot

    fake_gemini = MagicMock()
    fake_eval = types.SimpleNamespace(
        choices=[
            types.SimpleNamespace(
                message=types.SimpleNamespace(
                    parsed=chatbot.Evaluation(is_acceptable=True, feedback="ok")
                )
            )
        ]
    )
    fake_gemini.beta.chat.completions.parse.return_value = fake_eval
    monkeypatch.setattr(chatbot, "gemini", fake_gemini)
    result = chatbot.evaluate("reply", "msg", "hist")
    assert result.is_acceptable is True
    assert result.feedback == "ok"


def test_rerun(monkeypatch):
    import chatbot

    fake_openai = MagicMock()
    fake_openai.chat.completions.create.return_value.choices = [
        types.SimpleNamespace(message=types.SimpleNamespace(content="rerun reply"))
    ]
    monkeypatch.setattr(chatbot, "openai", fake_openai)
    reply = chatbot.rerun(
        "bad reply", "msg", [{"role": "user", "content": "hi"}], "bad feedback"
    )
    assert "rerun reply" == reply


def test_chat_accept(monkeypatch):
    import chatbot

    # Acceptable evaluation
    monkeypatch.setattr(chatbot, "get_openai_reply", lambda *a, **k: "good reply")
    monkeypatch.setattr(
        chatbot,
        "evaluate",
        lambda *a, **k: chatbot.Evaluation(is_acceptable=True, feedback="ok"),
    )
    result = chatbot.chat("hello", [])
    assert result == "good reply"


def test_chat_reject(monkeypatch):
    import chatbot

    # Not acceptable, triggers rerun
    monkeypatch.setattr(chatbot, "get_openai_reply", lambda *a, **k: "bad reply")
    monkeypatch.setattr(
        chatbot,
        "evaluate",
        lambda *a, **k: chatbot.Evaluation(is_acceptable=False, feedback="bad"),
    )
    monkeypatch.setattr(chatbot, "rerun", lambda *a, **k: "rerun reply")
    result = chatbot.chat("hello", [])
    assert result == "rerun reply"


def test_chat_pig_latin(monkeypatch):
    import chatbot

    # Test pig latin system prompt
    called = {}

    def fake_get_openai_reply(messages, system=None):
        called["system"] = system
        return "pig latin reply"

    monkeypatch.setattr(chatbot, "get_openai_reply", fake_get_openai_reply)
    monkeypatch.setattr(
        chatbot,
        "evaluate",
        lambda *a, **k: chatbot.Evaluation(is_acceptable=True, feedback="ok"),
    )
    chatbot.chat("patent", [])
    assert "pig latin" in called["system"]


def test_evaluation_model():
    import chatbot

    eval = chatbot.Evaluation(is_acceptable=False, feedback="bad")
    assert not eval.is_acceptable
    assert eval.feedback == "bad"
