.PHONY: build up down logs shell train-siamese train-classifier extract-crops test export-onnx

build:
	docker-compose build

up:
	docker-compose up -d

down:
	docker-compose down

logs:
	docker-compose logs -f

shell:
	docker-compose exec api bash

train-siamese:
	docker-compose exec api python train_optimized.py --model-version v2

train-classifier:
	docker-compose exec api python member3_classifier/train_classifier.py

extract-crops:
	docker-compose exec api python member3_classifier/extract_crops.py

test:
	docker-compose exec api pytest tests/

export-onnx:
	docker-compose exec api python scripts/export_onnx.py --model siamese --ckpt checkpoints/siamese_best.pth
