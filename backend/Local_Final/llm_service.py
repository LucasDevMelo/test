from vertexai.generative_models import GenerativeModel
import vertexai
from google.api_core.exceptions import GoogleAPICallError, RetryError, FailedPrecondition
from config import (
    VERTEX_PROJECT,
    VERTEX_LOCATION,
    DEFAULT_LLM_MODEL
)

# Initialize Vertex AI
vertexai.init(
    project=VERTEX_PROJECT,
    location=VERTEX_LOCATION,
)


def call_llm(prompt: str, model_name: str = DEFAULT_LLM_MODEL):
    """
    Calls Vertex AI Gemini models using the official GenerativeModel API.
    Includes basic error handling.
    """
    try:
        model = GenerativeModel(model_name)
        response = model.generate_content(prompt)
        return response.text

    except (GoogleAPICallError, RetryError) as api_err:
        # API call failed or timed out
        print(f"[ERROR] Vertex AI API call failed: {api_err}")
        return "Error: Unable to generate answer at the moment."

    except FailedPrecondition as precond_err:
        # Model or project not ready
        print(f"[ERROR] Vertex AI failed precondition: {precond_err}")
        return "Error: LLM model not ready. Please try again later."

    except Exception as e:
        # Catch all other exceptions
        print(f"[ERROR] Unexpected error: {e}")
        return "Error: Something went wrong during LLM call."
