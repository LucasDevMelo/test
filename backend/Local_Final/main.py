import re
import json
from query_service import search
from context_layer import build_context_layer, default_model as model
from llm_service import call_llm
from config import (
    CONFIDENCE_THRESHOLD,
    NO_CITATION_CONFIDENCE_CAP,
    HIGH_RISK_KEYWORDS,
    HUMAN_REQUEST_KEYWORDS,
)


def map_llm_intent(intent_text):
    intent_text = intent_text.lower()
    for std_intent, keywords in HIGH_RISK_KEYWORDS.items():
        if any(k in intent_text for k in keywords):
            return std_intent
    return intent_text  # fallback to raw text


def normalize_confidence(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, value))


def apply_escalation_rules(llm_json, query_text: str, has_context: bool):
    """
    Enforce deterministic escalation logic at system level

    Priority:
    1. User explicitly requests a human agent
    2. Policy-based high-risk intents
    3. Missing usable context (NOT missing citations)
    4. Low confidence
    """
    confidence = normalize_confidence(llm_json.get("confidence"))
    intent = map_llm_intent(llm_json.get("intent") or "")
    citations = llm_json.get("citations", [])

    escalate = False
    reason = None

    query_lower = query_text.lower()

    if any(k in query_lower for k in HUMAN_REQUEST_KEYWORDS):
        escalate = True
        reason = "user_requested_human"

    elif intent in HIGH_RISK_KEYWORDS:
        escalate = True
        reason = intent

    elif not has_context:
        escalate = True
        reason = "missing_relevant_docs"

    elif confidence < CONFIDENCE_THRESHOLD:
        escalate = True
        reason = "low_confidence"

    # Confidence cap when no citations are provided
    if not citations:
        confidence = min(confidence, NO_CITATION_CONFIDENCE_CAP)

    llm_json["confidence"] = confidence
    llm_json["escalate"] = escalate
    llm_json["escalation_reason"] = reason

    return llm_json


def parse_llm_json(raw_output):
    """
    Safely extract JSON from LLM Markdown output (```json ... ``` block)
    """
    match = re.search(r"```json\s*(\{.*?\})\s*```", raw_output, re.DOTALL)
    if match:
        json_str = match.group(1)
        try:
            return json.loads(json_str)
        except Exception as e:
            print("\nFailed to parse JSON inside ```json block```:", e)
    return None


def main():
    print("Welcome to the search system. Type 'exit' to quit.")

    while True:
        query_text = input("\nEnter your query: ")
        if query_text.lower() in ["exit", "quit"]:
            print("Exiting program.")
            break

        # Vector Search
        query_vector = model.encode(query_text, normalize_embeddings=True)
        results = search(query_vector, query_text)

        if not results:
            print("No similar results found.")
            continue

        # Build Context Layer
        context_output = build_context_layer(results, embedder=model)
        context_text = context_output["context_text"]
        has_context = context_output.get("has_context", False)

        # LLM Prompt
        prompt = f"""
You are an expert assistant.

# CONFIDENCE DEFINITION
Confidence is a number between 0 and 1 representing how well your answer is supported by the provided context.

# OUTPUT FORMAT
1. First: a clean Markdown answer for the user.
2. Then: a JSON object ONLY, following this schema:

{{
  "answer": "short answer summary",
  "steps": ["documents consulted to answer the question"],
  "citations": ["URLs used"],
  "confidence": 0.0,
  "escalate": false,
  "escalation_reason": null,
  "intent": "user intent"
}}

# RULES
- Confidence MUST be between 0 and 1.
- Use ONLY URLs from the context as citations.
- Do NOT reference internal document IDs or chunk IDs anywhere in the output.
- When mentioning a document, always include its full URL in the same sentence.
- Do NOT mention documents without including a URL.
- The citations field must contain only URLs that appear in the answer.
- Steps are a short summary of which documents were consulted, written for users.
- Steps must be factual and minimal.
- JSON must be valid and last.
- If no document directly answers the question, return an empty citations list.
- Do NOT cite generic documents unless the question is explicitly general.

# USER QUERY
{query_text}

# CONTEXT
{context_text}

Respond now.
"""

        raw_output = call_llm(prompt)
        print("\n--- LLM Raw Output ---\n")
        print(raw_output)

        # Parse JSON from LLM Output
        llm_json = parse_llm_json(raw_output)

        if llm_json:
            llm_json = apply_escalation_rules(
                llm_json,
                query_text=query_text,
                has_context=has_context
            )

            print("\n--- Final Structured Output ---\n")
            print(json.dumps(llm_json, indent=2, ensure_ascii=False))

        else:
            fallback = {
                "answer": "Unable to parse LLM output.",
                "steps": ["LLM output did not contain valid JSON block"],
                "citations": [],
                "confidence": 0.0,
                "escalate": True,
                "escalation_reason": "llm_output_parse_failed",
                "intent": "unknown"
            }
            print("\nLLM output could not be parsed. Escalating automatically.")
            print(json.dumps(fallback, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
