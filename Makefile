PYTHON    := python3
DATASET   := ds004504
DATA_RAW  := data/raw
DATA_PROC := data/processed
CKPT_DIR  := models/checkpoints
ONNX_DIR  := models/onnx
RESULTS   := results

.PHONY: setup download preprocess train export bench loso-bench feature-bench \
        download-parkinson preprocess-parkinson domain-shift test clean all

all: preprocess train export bench

setup:
	$(PYTHON) -m pip install -r requirements.txt

download:
	$(PYTHON) -m openneuro download --dataset $(DATASET) --target-dir $(DATA_RAW)

download-parkinson:
	$(PYTHON) -m openneuro download --dataset ds002778 --target-dir $(DATA_RAW)

preprocess:
	$(PYTHON) -m preprocess.pipeline --data-dir $(DATA_RAW) --out-dir $(DATA_PROC)

preprocess-parkinson:
	$(PYTHON) -m preprocess.pipeline --data-dir $(DATA_RAW)/ds002778 --out-dir data/processed_parkinson

train:
	$(PYTHON) -m models.train_sklearn --data-dir $(DATA_PROC) --out-dir $(CKPT_DIR)

export:
	$(PYTHON) -m infer.export_sklearn \
	    --model-dir $(CKPT_DIR) --out-dir $(ONNX_DIR) --data-dir $(DATA_PROC)

bench:
	$(PYTHON) -m bench.run --model-dir $(ONNX_DIR) --data-dir $(DATA_PROC) --out-dir $(RESULTS)

loso-bench:
	$(PYTHON) -m bench.loso_bench \
	    --data-dir $(DATA_PROC) --model-dir $(CKPT_DIR) --out-dir $(RESULTS)

feature-bench:
	$(PYTHON) -m bench.feature_bench --n-windows 500 --batch-size 64

domain-shift:
	$(PYTHON) -m bench.domain_shift \
	    --source-model $(CKPT_DIR)/svm_rbf/best_model.pkl \
	    --target-dir data/processed_parkinson \
	    --out-dir $(RESULTS)

test:
	$(PYTHON) -m pytest tests/ -v --tb=short

clean:
	rm -rf $(DATA_PROC) data/processed_parkinson \
	       $(CKPT_DIR) $(ONNX_DIR) \
	       $(RESULTS)/*.csv $(RESULTS)/*.json
