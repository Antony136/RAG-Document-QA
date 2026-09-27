import json

from ollama import chat


MODEL = "qwen2.5-coder:7b"


JUDGE_SYSTEM_PROMPT = """
You are an evaluator for a Retrieval-Augmented Generation (RAG) system.

You must evaluate the generated answer using ONLY:
1. The user's question.
2. The expected answer.
3. The retrieved context.
4. The generated answer.

Do not use outside knowledge.

Evaluate four dimensions.

1. Answer Correctness
How well does the generated answer match the expected answer?
- 5 = fully correct and complete
- 4 = mostly correct with minor omissions
- 3 = partially correct
- 2 = substantially incomplete or contains notable errors
- 1 = incorrect

2. Answer Relevance
How directly does the generated answer answer the user's question?
- 5 = directly and completely answers the question
- 4 = directly answers with minor unnecessary content
- 3 = partially answers the question
- 2 = mostly off-topic or incomplete
- 1 = does not answer the question

3. Faithfulness
Is every important factual claim in the generated answer supported by the retrieved context?
- 5 = all important claims are supported
- 4 = almost all claims are supported
- 3 = some claims are unsupported or unclear
- 2 = several important claims are unsupported
- 1 = answer is largely unsupported by the context

4. Context Relevance
How useful is the retrieved context for answering the question?
- 5 = context contains the necessary information clearly
- 4 = context is highly useful with minor irrelevant content
- 3 = context contains some useful information but also substantial irrelevant content
- 2 = context provides little useful information
- 1 = context does not contain useful information

Return ONLY valid JSON in exactly this structure:

{
  "correctness": {
    "score": 1,
    "reason": "..."
  },
  "relevance": {
    "score": 1,
    "reason": "..."
  },
  "faithfulness": {
    "score": 1,
    "reason": "..."
  },
  "context_relevance": {
    "score": 1,
    "reason": "..."
  }
}

Scores must be integers from 1 to 5.
Do not include markdown.
"""


def build_judge_prompt(
    question: str,
    expected_answer: str,
    context: str,
    generated_answer: str,
):
    return f"""
Question:
{question}

Expected answer:
{expected_answer}

Retrieved context:
{context}

Generated answer:
{generated_answer}
""".strip()


def _extract_json(content: str):
    content = content.strip()

    if content.startswith("```"):
        lines = content.splitlines()

        if lines and lines[0].startswith("```"):
            lines = lines[1:]

        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]

        content = "\n".join(lines).strip()

    return json.loads(content)


def _validate_score(value, name):
    if not isinstance(value, int):
        raise ValueError(
            f"{name} score must be an integer."
        )

    if value < 1 or value > 5:
        raise ValueError(
            f"{name} score must be between 1 and 5."
        )


def validate_judgment(judgment):
    required_dimensions = (
        "correctness",
        "relevance",
        "faithfulness",
        "context_relevance",
    )

    for dimension in required_dimensions:
        if dimension not in judgment:
            raise ValueError(
                f"Missing judgment dimension: {dimension}"
            )

        value = judgment[dimension]

        if not isinstance(value, dict):
            raise ValueError(
                f"{dimension} must be an object."
            )

        if "score" not in value:
            raise ValueError(
                f"Missing score for {dimension}."
            )

        if "reason" not in value:
            raise ValueError(
                f"Missing reason for {dimension}."
            )

        _validate_score(
            value["score"],
            dimension,
        )

        if not isinstance(value["reason"], str):
            raise ValueError(
                f"{dimension} reason must be a string."
            )

    return judgment


def judge_answer(
    question: str,
    expected_answer: str,
    context: str,
    generated_answer: str,
):
    prompt = build_judge_prompt(
        question=question,
        expected_answer=expected_answer,
        context=context,
        generated_answer=generated_answer,
    )

    response = chat(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": JUDGE_SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        options={
            "temperature": 0.0,
            "num_predict": 700,
        },
    )

    content = response.message.content

    judgment = _extract_json(content)

    return validate_judgment(judgment)