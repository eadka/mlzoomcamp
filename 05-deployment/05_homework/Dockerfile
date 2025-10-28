FROM agrigorev/zoomcamp-model:2025

# Install uv (package manager)
RUN pip install uv

WORKDIR /app

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install dependencies
RUN uv sync --locked

# Copy your FastAPI app
COPY predict_pipeline.py .

EXPOSE 9696

# Run FastAPI app with uvicorn
ENTRYPOINT ["uv", "run", "uvicorn", "predict_pipeline:app", "--host", "0.0.0.0", "--port", "9696"]
