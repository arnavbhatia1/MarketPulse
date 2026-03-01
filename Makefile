.PHONY: setup ingest label train evaluate run all clean pipeline test

setup:
	pip install -r requirements.txt

ingest:
	python scripts/ingest.py --days 7

label:
	python scripts/label.py

train:
	python scripts/train.py --source programmatic

evaluate:
	python scripts/evaluate.py

run:
	streamlit run app/streamlit_app.py

all: ingest label train evaluate run

pipeline:
	python scripts/run_pipeline.py

clean:
	rm -rf data/raw/*
	rm -rf data/labeled/*
	rm -rf data/models/*.pkl
	rm -rf data/models/*.json

test:
	pytest tests/ -v
