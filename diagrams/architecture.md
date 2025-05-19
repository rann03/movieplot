flowchart TD
    %% Nodes
    A1[Browser: Streamlit UI]
    A2[streamlit_app.py]
    A3[POST /recommend to FastAPI]

    B1[main.py]
    B2[api/recommend_movies.py]
    B3[services/movie_recommender.py]
    B4[schemas/movie.py]
    B5[core/config.py]
    B6[.env / Settings]

    C1["OpenRouter API (Qwen)"]

    D1[scripts/model_accuracy_eval.py]
    D2[scripts/test_data/loader.py]
    D3[MLflow Tracking]
    D4[movie CSV files]

    E1[ci.yml - tests, lint, build]
    E2[model_accuracy.yml - manual run]

    %% Groups
    subgraph User
        A1
    end

    subgraph "Frontend (Streamlit)"
        A2
        A3
    end

    subgraph "Backend (FastAPI)"
        B1
        B2
        B3
        B4
        B5
        B6
    end

    subgraph "MLflow Evaluation"
        D1
        D2
        D3
        D4
    end

    subgraph "CI/CD (GitHub Actions)"
        E1
        E2
    end

    %% Connections
    A1 --> A2 --> A3 --> B1
    B1 --> B2 --> B3
    B2 --> B4
    B1 --> B5 --> B6
    B3 --> C1

    D1 --> D2 --> D4
    D1 --> C1
    D1 --> D3

    E2 --> D1
    E2 --> D3