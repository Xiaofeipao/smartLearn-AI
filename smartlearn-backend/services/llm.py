import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

SYSTEM_PROMPT = (
    "You answer messages only from the supplied PDF text. "
    "Cite factual claims with [Page X]. "
    "If the answer is not in the PDF, say that the document does not provide enough information. "
    "Never invent a page number."
)


def answer_from_pages(
    pages: list[dict],
    message: str,
    history: list[dict] | None = None,
) -> str:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not configured")

    document_text = "\n\n".join(
        f"### [Page {page['page']}]\n{page['text']}"
        for page in pages
        if page["text"]
    )

    # First user message carries the full PDF text
    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"PDF text:\n{document_text}"},
    ]

    # Replay conversation history so the LLM has full context
    if history:
        for turn in history:
            messages.append({"role": "user", "content": turn["question"]})
            messages.append({"role": "assistant", "content": turn["answer"]})

    # Current question
    messages.append({"role": "user", "content": message})

    client = OpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
    )
    response = client.chat.completions.create(
        model=os.getenv("OPENROUTER_MODEL", "google/gemma-4-26b-a4b-it:free"),
        temperature=0.0,
        messages=messages,
    )
    return response.choices[0].message.content or ""
