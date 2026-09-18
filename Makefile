.PHONY: install test smoke demo full api dashboard docker clean

install:
	pip install -r requirements.txt

test:
	pytest tests/ -q

smoke:
	python run.py smoke

demo:
	python run.py demo

full:
	python run.py full

api:
	python run.py api

dashboard:
	python run.py dashboard

docker:
	docker compose up --build

clean:
	rm -rf artifacts/*.json artifacts/*.png artifacts/SUMMARY.md artifacts/*.log
	find . -name __pycache__ -type d -exec rm -rf {} +
