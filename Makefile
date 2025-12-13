build:
	./build.sh

run:
	./run.sh

clean:
	./clean.sh

help:
	@echo "Makefile commands:"
	@echo "  build   - Build the Docker image"
	@echo "  run     - Run the Docker container"
	@echo "  clean   - Clean data and model directories"
	@echo "  help    - Show this help message"