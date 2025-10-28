PORT ?= 8000

.PHONY: dev index test

dev:
	uv run uvicorn src.api:app --host 0.0.0.0 --port $(PORT) --reload
	
index:
	uv run python -m src.ingest
