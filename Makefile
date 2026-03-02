.PHONY: setup ingest label train evaluate run all clean pipeline test

setup:
	pip install -r requirements.txt

ingest:
	python3 scripts/ingest.py --days 7

label:
	python3 scripts/label.py

train:
	python3 scripts/train.py --source programmatic

evaluate:
	python3 scripts/evaluate.py

run:
	python3 -m streamlit run app/MarketPulse.py

all: ingest label train evaluate run

pipeline:
	python3 scripts/run_pipeline.py

clean:
	rm -rf data/raw/*
	rm -rf data/labeled/*
	rm -rf data/models/*.pkl
	rm -rf data/models/*.json

test:
	pytest tests/ -v
