import json
import re
from typing import List, Tuple


def parse_suggestions_from_json(content: str) -> List[dict]:
    """
    Extracts movie title + description pairs from LLM response.

    Expected format:
    {
        "suggestions": [
            {"title": "X", "description": "Y"},
            ...
        ]
    }

    Parameters
    ----------
    content : str
        Raw LLM response text.

    Returns
    -------
    List[dict]
        A list of dicts with 'title' and 'description' keys.
    """
    try:
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if match:
            parsed = json.loads(match.group())
            if "suggestions" in parsed and isinstance(parsed["suggestions"], list):
                return [
                    {
                        "title": item.get("title", "Untitled").strip(),
                        "description": item.get(
                            "description", "No description."
                        ).strip(),
                    }
                    for item in parsed["suggestions"][:5]
                ]
    except Exception:
        pass

    return []


def parse_scores_from_json(
    content: str, suggestions: List[str]
) -> List[Tuple[str, float]]:
    """
    Parses a list of scores from the LLM response and normalizes them.

    Parameters
    ----------
    content : str
        The raw LLM response.
    suggestions : List[str]
        Titles that the scores should align with.

    Returns
    -------
    List[Tuple[str, float]]
        Movie titles and their normalized score (0.0 to 1.0).
    """
    try:
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if match:
            parsed = json.loads(match.group())
            raw_scores = parsed.get("scores", [])
            if isinstance(raw_scores, list) and len(raw_scores) == len(suggestions):
                return [
                    (title, round((int(score) - 1) / 4, 2))
                    for title, score in zip(suggestions, raw_scores)
                ]
    except Exception:
        pass

    return [(title, 0.0) for title in suggestions]
