format:
	isort -rc -y .
	black -l 79 .

test:
	pytest -n auto -vvv

envupdate:
	conda env update -f environment.yml

envupdategpu:
	conda env update -f environment.yml
	./scripts.py/install_jax_gpu.sh
