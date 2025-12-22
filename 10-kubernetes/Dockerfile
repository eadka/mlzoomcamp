FROM python:3.12-slim

#RUN pip install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Add the virtual environment's bin directory to the PATH so Python tools work globally
ENV PATH="/code/.venv/bin:$PATH"

WORKDIR /app

COPY ".python-version" "pyproject.toml" "./"
RUN uv sync

COPY "app.py" "clothing_classifier_mobilenet_v2_latest.onnx" "./"

EXPOSE 8080

ENTRYPOINT ["uv", "run", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]