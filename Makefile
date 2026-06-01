PYTHON    := python3
DATASET   := ds004504
DATA_RAW  := data/raw
DATA_PROC := data/processed
CKPT_DIR  := models/checkpoints
ONNX_DIR  := models/onnx
RESULTS   := results

.PHONY: setup download preprocess train export bench test clean all

all: preprocess train export bench

setup:
	$(PYTHON) -m pip install -r requirements.txt

download:
	$(PYTHON) -m openneuro download --dataset $(DATASET) --target-dir $(DATA_RAW)

preprocess:
	$(PYTHON) -m preprocess.pipeline --data-dir $(DATA_RAW)/$(DATASET) --out-dir $(DATA_PROC)

train:
	$(PYTHON) -m models.train_sklearn --data-dir $(DATA_PROC) --out-dir $(CKPT_DIR)

export:
	$(PYTHON) -m infer.export_sklearn --model-dir $(CKPT_DIR) --out-dir $(ONNX_DIR)

bench:
	$(PYTHON) -m bench.run --model-dir $(ONNX_DIR) --data-dir $(DATA_PROC) --out-dir $(RESULTS)

test:
	$(PYTHON) -m pytest tests/ -v --tb=short

clean:
	rm -rf $(DATA_PROC) $(CKPT_DIR) $(ONNX_DIR) $(RESULTS)/*.csv $(RESULTS)/*.json
