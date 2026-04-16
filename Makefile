docker-sim:
	docker build -t visionpilot-sim -f docker/Dockerfile .
	docker run --rm -v "$(PWD)":/workspace -w /workspace visionpilot-sim --help

.PHONY: docker-sim
