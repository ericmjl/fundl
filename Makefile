make check:
	black -l 79 src/

make test:
	pytest -n auto
