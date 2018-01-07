.PHONY: deps
deps:
	pip3 install --upgrade pip
	pip3 install tensorflow dm-sonnet

.PHONY: check
check:
	pycodestyle --max-line-length=100 dnc
	pylint -E dnc

.PHONY: test
test: check
	python3 -m unittest discover -v unittests

.PHONY: viz
viz:
	rm -rf ./graphs/*
	python3 visualize_graph.py
	tensorboard --host=localhost --logdir=graphs

.PHONY: clean
clean:
	rm -rf ./graphs ./model* ./logs ./__pycache__
	find . -name '*.pyc' -delete

.PHONY: babi
babi:
	python3 babi/train.py
