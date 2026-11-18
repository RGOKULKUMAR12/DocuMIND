import os
from functools import lru_cache
from typing import List, Dict, Any, Union
from dotenv import load_dotenv
from google import genai
from google.genai.errors import APIError

# Load environment variables from .env
load_dotenv()

DEFAULT_MODEL = "gemini-3.1-flash-lite"


@lru_cache(maxsize=1)
def get_gemini_client() -> genai.Client:
    """
    Initializes and returns the official Google GenAI client using the API key from .env.
    """
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError(
            "Gemini API key not found. Please set GEMINI_API_KEY in your .env file."
        )
    return genai.Client(api_key=api_key)


def format_context(context: Union[str, List[Dict[str, Any]]]) -> str:
    """
    Normalizes context into a clean string.
    Accepts either a string or a list of retrieved chunk dictionaries.
    """
    if not context:
        return ""

    if isinstance(context, str):
        return context.strip()

    formatted_parts = []
    for chunk in context:
        if isinstance(chunk, dict):
            content = chunk.get("content", "").strip()
            source = chunk.get("source", "Document")
            page = chunk.get("page_number", "?")
            formatted_parts.append(f"[Source: {source}, Page: {page}]\n{content}")
        else:
            formatted_parts.append(str(chunk).strip())

    return "\n\n".join(formatted_parts).strip()


def generate_answer(
    question: str,
    context: Union[str, List[Dict[str, Any]]],
    model_name: str = DEFAULT_MODEL
) -> str:
    """
    Generates a context-grounded answer to a user question using Gemini.
    Instructs the model to answer strictly using the provided context.

    Parameters:
        question: The user's query or question.
        context: Retrieved text chunks (as a string or list of chunk dicts).
        model_name: Gemini model identifier (default: 'gemini-3.1-flash-lite').

    Returns:
        str: Grounded answer or notification if information is unavailable.
    """
    if not question or not question.strip():
        return "Please provide a valid question."

    context_str = format_context(context)
    if not context_str:
        return "The provided documents do not contain sufficient information to answer this question."

    prompt = f"""Use only this document context to answer the question.
Be direct and concise. If the context contains a relevant definition or explanation,
provide it even when the exact wording is spread across the excerpts.
Only say that the context is insufficient when it has no relevant answer.

Context:
{context_str[:3500]}

Question: {question.strip()}

Answer:"""

    try:
        client = get_gemini_client()
        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config={
                "temperature": 0.2,
                "max_output_tokens": 192,
            }
        )

        if response and response.text:
            return response.text.strip()
        return "No answer could be generated from the model."

    except ValueError as ve:
        # Raised when GEMINI_API_KEY is not configured
        return f"Configuration Error: {str(ve)}"
    except APIError as ae:
        # Handles Google GenAI API errors (quota, invalid key, service errors)
        return f"Gemini API Error: {str(ae)}"
    except Exception as e:
        return f"Unexpected Error: {str(e)}"
