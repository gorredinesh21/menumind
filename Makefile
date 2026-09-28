.PHONY: run test vet warm docker deploy clean

run:
	go run .

test:
	go test ./...

vet:
	go vet ./...

warm: ## re-embed the corpus (requires HF_TOKEN)
	go run ./cmd/warm -workers 10

docker:
	docker build -t menumind .

deploy: ## Cloud Run (requires gcloud auth + HF_TOKEN in env)
	gcloud run deploy menumind \
		--project personal-project-dg21 --region us-central1 \
		--source . --allow-unauthenticated \
		--set-env-vars HF_TOKEN=$${HF_TOKEN} \
		--min-instances=0 --max-instances=2 --memory=512Mi --cpu=1 \
		--timeout=120

clean:
	rm -rf bin
