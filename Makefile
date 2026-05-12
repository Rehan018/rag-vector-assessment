install:
	pip install -r requirements.txt

test:
	pytest

benchmark:
	python main.py

run:
	python main.py