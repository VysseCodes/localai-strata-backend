.DEFAULT_GOAL := install

.PHONY: install
install:
	bash install.sh

.PHONY: strata
strata: install

.PHONY: run
run: install
	@echo "Running strata..."
	bash run.sh
	@echo "strata run."

.PHONY: test
test: install
	@echo "Testing strata..."
	bash test.sh
	@echo "strata tested."

.PHONY: protogen-clean
protogen-clean:
	$(RM) backend_pb2_grpc.py backend_pb2.py

.PHONY: clean
clean: protogen-clean
	rm -rf venv __pycache__
