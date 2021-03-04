.PHONY: container
container:
	docker-compose up --build notebook

.PHONY: local
local:
	pip install -r notebook/requirements.txt
	jupyter notebook --no-browser --ip=0.0.0.0 --port=8080
