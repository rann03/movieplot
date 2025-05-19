
FROM python:3.11-slim

WORKDIR /src

COPY . .

RUN pip install --upgrade pip && \
    pip install --no-cache-dir streamlit requests python-dotenv

EXPOSE 8501

CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]


