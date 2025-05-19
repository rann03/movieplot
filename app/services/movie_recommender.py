import logging
from typing import List, Tuple

import requests

from app.core.config import SETTINGS
from app.schemas.movie import MovieSuggestionScored
from app.utils.parsers import parse_scores_from_json, parse_suggestions_from_json

logger = logging.getLogger(__name__)


class MovieRecommender:
    """Service class for generating and scoring movie title suggestions via LLM."""

    def get_movie_titles_from_llm(
        self, plot: str, model: str = "anthropic/claude-2"
    ) -> List[dict]:
        logger.info(f"Generating movie titles using model: {model}")

        headers = {
            "Authorization": f"Bearer {SETTINGS.API_KEY}",
            "Content-Type": "application/json",
        }

        prompt = (
            "You are a movie expert. Return 5 REAL MOVIES titles that best match the plot bellow:\n"
            "Each suggestion must include a short description of the movie.\n\n"
            f"Plot:\n{plot}\n\n"
            "Respond ONLY in this JSON format:\n"
            '{ "suggestions": [\n'
            '  {"title": "Title 1", "description": "Description 1"},\n'
            '  {"title": "Title 2", "description": "Description 2"},\n'
            '  {"title": "Title 3", "description": "Description 3"},\n'
            '  {"title": "Title 4", "description": "Description 4"},\n'
            '  {"title": "Title 5", "description": "Description 5"}\n]}'
        )

        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": "You are a movie title generator."},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": 500,
            "temperature": 0.1,
        }

        try:
            response = requests.post(
                url=SETTINGS.LLM_API_URL, headers=headers, json=payload, timeout=15
            )

            logger.info(f"LLM API responded with status {response.status_code}")
            try:
                logger.debug(f"Raw response text: {response.text}")
                json_data = response.json()
                logger.debug(f"Parsed response JSON: {json_data}")
            except Exception as json_err:
                logger.error("Failed to parse LLM response as JSON.")
                raise

            if response.status_code != 200:
                logger.error(f"LLM API returned non-200 status: {response.status_code}")
                response.raise_for_status()

            content = json_data["choices"][0]["message"]["content"]
            logger.debug(f"LLM suggestion response content: {content}")
            return parse_suggestions_from_json(content)

        except Exception:
            logger.exception("Failed to get movie title suggestions from LLM.")
            return [
                {"title": "Fallback Movie 1", "description": "No description."},
                {"title": "Fallback Movie 2", "description": "No description."},
                {"title": "Fallback Movie 3", "description": "No description."},
                {"title": "Fallback Movie 4", "description": "No description."},
                {"title": "Fallback Movie 5", "description": "No description."},
            ]

    def evaluate_suggestions(
        self, plot: str, suggestions: List[dict], model: str = "anthropic/claude-2"
    ) -> List[Tuple[str, float]]:
        logger.info("Scoring movie suggestions using LLM.")

        prompt = (
            "You are a movie critic. Evaluate how well each of the following suggested movie titles + description pairs "
            "match the plot below. Respond in JSON:\n"
            '{"scores": [1, 2, 3, 4, 5]}\n\n'
            f"Plot:\n{plot}\n\n"
            f"Suggestions:\n"
            + "\n".join(
                f"{i+1}. {s['title']} — {s['description']}"
                for i, s in enumerate(suggestions)
            )
        )

        headers = {
            "Authorization": f"Bearer {SETTINGS.API_KEY}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": "You evaluate how well movie title+description pairs match a given plot.",
                },
                {"role": "user", "content": prompt},
            ],
            "max_tokens": 150,
            "temperature": 0.3,
        }

        try:
            response = requests.post(
                url=SETTINGS.LLM_API_URL, headers=headers, json=payload, timeout=15
            )

            if response.status_code != 200:
                logger.error(
                    f"LLM scoring API error: Status {response.status_code}, Response: {response.text}"
                )
                response.raise_for_status()

            content = response.json()["choices"][0]["message"]["content"]
            logger.debug(f"LLM score response: {content}")

            titles = [s["title"] for s in suggestions]
            return parse_scores_from_json(content, titles)

        except Exception as e:
            logger.exception("Failed to evaluate movie suggestions.")
            return [(s["title"], 0.0) for s in suggestions]

    def suggest_and_score_movies(
        self, plot: str, model: str
    ) -> List[MovieSuggestionScored]:
        logger.info(f"Starting movie recommendation process with model: {model}")
        suggestions = self.get_movie_titles_from_llm(plot, model)
        logger.info(f"Received {len(suggestions)} suggestions from LLM.")

        scored_pairs = self.evaluate_suggestions(plot, suggestions, model)
        logger.info("Scoring completed.")

        scored_suggestions = [
            MovieSuggestionScored(
                title=title,
                description=next(
                    (s["description"] for s in suggestions if s["title"] == title),
                    "No description.",
                ),
                score=score,
            )
            for title, score in scored_pairs
        ]

        logger.info("Final scored suggestions assembled.")
        return sorted(scored_suggestions, key=lambda x: x.score, reverse=True)
