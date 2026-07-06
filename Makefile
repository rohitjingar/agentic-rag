.PHONY: up down migrate models test lint serve

up:
	docker compose up -d --wait

down:
	docker compose down

migrate:
	uv run rag-migrate

models:
	./scripts/bootstrap_models.sh

test:
	uv run pytest -q

lint:
	uv run ruff check . && uv run ruff format --check .

serve:
	uv run uvicorn rag.api.app:app --reload
