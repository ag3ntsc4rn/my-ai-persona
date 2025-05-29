import os
import sys
from dotenv import load_dotenv
from openai import OpenAI
from pypdf import PdfReader
import gradio as gr
from pydantic import BaseModel

# --- Load environment and initialize clients ---
load_dotenv(override=True)
# --- Configuration ---
DATA_DIR = os.getenv("PROFILE_DATA_DIR", "data")
PROFILE_PDF = os.path.join(DATA_DIR, os.getenv("PROFILE_PDF", "Profile.pdf"))
SUMMARY_TXT = os.path.join(DATA_DIR, os.getenv("SUMMARY_TXT", "summary.txt"))
DEFAULT_NAME = os.getenv("PROFILE_NAME", "Ag3nt Sc4rn")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_BASE_URL = os.getenv(
    "GOOGLE_API_BASE_URL",
    "https://generativelanguage.googleapis.com/v1beta/openai/",
)


openai = OpenAI(api_key=OPENAI_API_KEY)
gemini = OpenAI(api_key=GEMINI_API_KEY, base_url=GEMINI_BASE_URL)


# --- Utility functions ---
def load_pdf_text(pdf_path):
    reader = PdfReader(pdf_path)
    return "".join(page.extract_text() or "" for page in reader.pages)


def load_text_file(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


# --- Load profile data ---
name = DEFAULT_NAME
linkedin = load_pdf_text(PROFILE_PDF)
summary = load_text_file(SUMMARY_TXT)


# --- Prompt builders ---
def build_system_prompt(name, summary, linkedin):
    return (
        f"You are acting as {name}. You are answering questions on {name}'s website, "
        f"particularly questions related to {name}'s career, background, skills and experience. "
        f"Your responsibility is to represent {name} for interactions on the website as faithfully as possible. "
        f"You are given a summary of {name}'s background and LinkedIn profile which you can use to answer questions. "
        f"Be professional and engaging, as if talking to a potential client or future employer who came across the website. "
        f"If you don't know the answer, say so.\n\n"
        f"## Summary:\n{summary}\n\n"
        f"## LinkedIn Profile:\n{linkedin}\n\n"
        f"With this context, please chat with the user, always staying in character as {name}."
    )


def build_evaluator_system_prompt(name, summary, linkedin):
    return (
        f"You are an evaluator that decides whether a response to a question is acceptable. "
        f"You are provided with a conversation between a User and an Agent. Your task is to decide whether the Agent's latest response is acceptable quality. "
        f"The Agent is playing the role of {name} and is representing {name} on their website. "
        f"The Agent has been instructed to be professional and engaging, as if talking to a potential client or future employer who came across the website. "
        f"The Agent has been provided with context on {name} in the form of their summary and LinkedIn details. Here's the information:\n\n"
        f"## Summary:\n{summary}\n\n"
        f"## LinkedIn Profile:\n{linkedin}\n\n"
        f"With this context, please evaluate the latest response, replying with whether the response is acceptable and your feedback."
    )


def evaluator_user_prompt(reply, message, history):
    return (
        f"Here's the conversation between the User and the Agent: \n\n{history}\n\n"
        f"Here's the latest message from the User: \n\n{message}\n\n"
        f"Here's the latest response from the Agent: \n\n{reply}\n\n"
        f"Please evaluate the response, replying with whether it is acceptable and your feedback."
    )


system_prompt = build_system_prompt(name, summary, linkedin)
evaluator_system_prompt = build_evaluator_system_prompt(name, summary, linkedin)


# --- Pydantic model for evaluation ---
class Evaluation(BaseModel):
    is_acceptable: bool
    feedback: str


# --- Core LLM functions ---
def get_openai_reply(messages, system=None):
    if system:
        messages = [{"role": "system", "content": system}] + messages
    return (
        openai.chat.completions.create(model=OPENAI_MODEL, messages=messages)
        .choices[0]
        .message.content
    )


def evaluate(reply, message, history) -> Evaluation:
    messages = [
        {"role": "system", "content": evaluator_system_prompt},
        {"role": "user", "content": evaluator_user_prompt(reply, message, history)},
    ]
    response = gemini.beta.chat.completions.parse(
        model=GEMINI_MODEL, messages=messages, response_format=Evaluation
    )
    return response.choices[0].message.parsed


def rerun(reply, message, history, feedback):
    updated_system_prompt = (
        system_prompt
        + "\n\n## Previous answer rejected\nYou just tried to reply, but the quality control rejected your reply\n"
        f"## Your attempted answer:\n{reply}\n\n"
        f"## Reason for rejection:\n{feedback}\n\n"
    )
    messages = history + [{"role": "user", "content": message}]
    return get_openai_reply(messages, system=updated_system_prompt)


# --- Chat function with evaluation and rerun ---
def chat(message, history):
    system = system_prompt
    if "patent" in message:
        system += (
            "\n\nEverything in your reply needs to be in pig latin - "
            "it is mandatory that you respond only and entirely in pig latin"
        )
    messages = history + [{"role": "user", "content": message}]
    reply = get_openai_reply(messages, system=system)

    evaluation = evaluate(reply, message, history)
    if evaluation.is_acceptable:
        print("Passed evaluation - returning reply")
        return reply
    else:
        print("Failed evaluation - retrying")
        print(evaluation.feedback)
        return rerun(reply, message, history, evaluation.feedback)


# --- Gradio interface ---
gr.ChatInterface(chat, type="messages").launch()
