FROM python:3.10-slim

WORKDIR /app

RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir "numpy<2.0" streamlit pandas catboost scikit-learn requests plotly

COPY . .

RUN mkdir -p .streamlit && touch .streamlit/secrets.toml

EXPOSE 8501

CMD ["python", "-m", "streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
