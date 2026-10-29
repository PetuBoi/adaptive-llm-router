FROM ghcr.io/astral-sh/uv:0.13.1 AS uv
FROM python:3.13-slim
COPY --from=uv /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy PYTHONUNBUFFERED=1
WORKDIR /app
COPY . .
RUN uv sync --locked --no-dev && groupadd -g 10001 app && useradd -u 10001 -g app app \
    && mkdir -p data/runtime artifacts && chown -R app:app /app
USER app
EXPOSE 8001 8502
CMD ["uv", "run", "--no-sync", "uvicorn", "routelab.api.app:app", "--host", "0.0.0.0", "--port", "8001"]
