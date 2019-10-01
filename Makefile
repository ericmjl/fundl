make format:
	isort -rc -y .
	black -l 79 .

make test:
	pytest -n auto -vvv
