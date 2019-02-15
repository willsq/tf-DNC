.PHONY: deps
deps:
	pip3 install --upgrade pip
	pip3 install -r requirements.txt

.PHONY: check
check:
	pyflakes unittests dnc
	pycodestyle --max-line-length=100 unittests dnc
	pylint -j2 unittests dnc

.PHONY: test
test:
	pytest -s -v --cov-report=html:htmlcov --cov-report=term --cov dnc unittests

.PHONY: viz
viz:
	rm -rf ./graphs/*
	python3 visualize_graph.py
	tensorboard --host=localhost --logdir=graphs

.PHONY: clean
clean:
	rm -rf graphs ./model* logs __pycache__ .pytest_cache htmlcov exp
	find . -name '*.pyc' -delete

.PHONY: babi
babi:
	python3 babi/train.py
