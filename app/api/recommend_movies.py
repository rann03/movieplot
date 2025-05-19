"""
Routes for movie recommendation using a language model.

This module defines the /recommend endpoint that accepts a movie plot,
generates suggestions using an LLM, evaluates them, and returns
scored movie recommendations.
"""

import logging

from fastapi import APIRouter, Depends, HTTPException, status

from app.schemas.movie import MovieResponseScored, PlotInput
from app.services.movie_recommender import MovieRecommender

router = APIRouter()
logger = logging.getLogger(__name__)


def get_recommender():
    """Dependency injection for the MovieRecommender service."""
    return MovieRecommender()


@router.post("/recommend", response_model=MovieResponseScored)
def recommend_movies(
    payload: PlotInput, recommender: MovieRecommender = Depends(get_recommender)
):
    """
    Recommend 5 movies based on a given plot and return their match scores.

    This endpoint uses an LLM to generate movie titles and then score them
    based on how well they match the input plot.

    Parameters
    ----------
    payload : PlotInput
        The plot description provided by the user.

    Returns
    -------
    MovieResponseScored
        A list of 5 movie suggestions with normalized match scores.
    """
    try:
        scored_suggestions = recommender.suggest_and_score_movies(
            payload.plot, payload.model
        )
        return MovieResponseScored(suggestions=scored_suggestions)

    except Exception as e:
        logger.error(f"Error in /recommend: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate movie recommendations.",
        )
