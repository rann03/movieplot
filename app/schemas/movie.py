"""
Schemas for movie recommendation API.

This module defines the Pydantic models used for request and response validation
in the movie recommendation endpoint. It includes models for the input plot,
scored movie suggestions, and the full response containing all recommendations.
"""

from typing import List

from pydantic import BaseModel


class PlotInput(BaseModel):
    """
    Request model for inputting a movie plot.

    Attributes
    ----------
    plot : str
        The text description of the plot to base movie suggestions on.
    model: str
        The LLM model we will ask
    """

    plot: str
    model: str = "anthropic/claude-2"


class MovieSuggestionScored(BaseModel):
    """
    Model representing a single movie suggestion with an evaluation score.

    Attributes
    ----------
    title : str
        The title of the suggested movie.
    description : str
        A short description or justification of the suggestion.
    score : float
        A normalized score (0.0 to 1.0) indicating how well the suggestion matches the plot.
    """

    title: str
    description: str
    score: float


class MovieResponseScored(BaseModel):
    """
    Response model containing a list of scored movie suggestions.

    Attributes
    ----------
    suggestions : List[MovieSuggestionScored]
        A list of movie suggestions with match scores.
    """

    suggestions: List[MovieSuggestionScored]
