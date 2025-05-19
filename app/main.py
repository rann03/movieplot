"""
Main entry point for the FastAPI Movie Recommender API.

This module initializes the FastAPI application, configures metadata,
adds the health check endpoint, and includes the recommendation router.
"""

import logging

from fastapi import FastAPI

from app.api.recommend_movies import router as recommend_router

# Set logging level globally
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)

app = FastAPI(
    title="LLM Movie Recommender",
    description="An API that uses a language model to suggest and evaluate movie titles based on a plot description.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)


@app.get("/health", tags=["Health"])
def health_check():
    """
    Health check endpoint to confirm the API is running.

    Returns
    -------
    dict
        A simple status message.
    """
    return {"status": "ok"}


# Include the movie recommendation routes
app.include_router(recommend_router, tags=["Movie Recommendations"])
