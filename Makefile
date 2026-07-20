.PHONY: install test smoke validate
CHECKPOINT ?= results/TRAIN_RUN/checkpoints/best.pt
install:
	python -m pip install -e ".[rag,analysis,dev]"
test:
	python -m pytest -m "not gpu"
smoke:
	python scripts/run_synthetic.py --config configs/core_smoke.yaml --checkpoint "$(CHECKPOINT)" --smoke
validate:
	python scripts/validate_results.py --results results
