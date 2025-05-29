# chatbot.py

A chatbot application that acts as a persona (e.g., for a personal website) using OpenAI and Gemini LLMs.  
It loads profile data from PDF and text files, builds context-rich prompts, evaluates LLM responses for quality,  
and provides a Gradio chat interface.

---

## Features

- Loads persona data from environment-configured files.
- Uses **OpenAI** for main chat responses.
- Uses **Gemini** for automated response evaluation.
- Automatically retries responses if evaluation fails.
- Supports special prompt behavior (e.g., pig latin for "patent" queries).
- Gradio-based web chat interface.

---

## Environment Variables

| Variable                | Description                                         | Default/Required                                  |
|-------------------------|-----------------------------------------------------|---------------------------------------------------|
| `PROFILE_DATA_DIR`      | Directory containing profile data files             | `data`                                            |
| `PROFILE_PDF`           | Filename for the LinkedIn profile PDF               | `Profile.pdf`                                     |
| `SUMMARY_TXT`           | Filename for the summary text file                  | `summary.txt`                                     |
| `PROFILE_NAME`          | Persona name                                        | `Ag3nt Sc4rn`                                     |
| `OPENAI_MODEL`          | OpenAI model name                                   | `gpt-4o-mini`                                     |
| `GEMINI_MODEL`          | Gemini model name                                   | `gemini-2.0-flash`                                |
| `GOOGLE_API_KEY`        | Gemini API key                                      | **required**                                      |
| `GOOGLE_API_BASE_URL`   | Gemini API base URL                                 | `https://generativelanguage.googleapis.com/v1beta/openai/` |
| `OPENAI_API_KEY`        | OpenAI API key                                      | **required**                                      |
---

## Configuration & Usage

1. **Install dependencies:**
    ```bash
    pip install openai python-dotenv pypdf gradio pydantic
    ```

2. **Prepare your data directory** (default: `./data`) with:
    - `Profile.pdf`: Your LinkedIn profile exported as PDF.
    - `summary.txt`: A summary of your background.

3. **Set environment variables** (e.g., in a `.env` file):
    ```env
    PROFILE_DATA_DIR=data
    PROFILE_PDF=Profile.pdf
    SUMMARY_TXT=summary.txt
    PROFILE_NAME=Your Name
    OPENAI_MODEL=gpt-4o-mini
    GEMINI_MODEL=gemini-2.0-flash
    GOOGLE_API_KEY=your-gemini-api-key
    GOOGLE_API_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
    OPENAI_API_KEY=openai-api-key
    ```

4. **Run the chatbot:**
    ```bash
    python chatbot.py
    ```

5. **Access the Gradio interface** in your browser as instructed in the terminal.

---

## Notes

- The chatbot will impersonate the configured persona and answer questions based on the provided summary and LinkedIn profile.
- Responses are automatically evaluated for quality; if rejected, the chatbot will retry with feedback.
- For questions containing "patent", the chatbot will reply in pig latin.
