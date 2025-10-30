# Simple Makefile for Maven build without tests
.PHONY: build clean package help

# Maven wrapper
MVN = ./mvnw

# Default target
all: package

# Build the project (clean and package without tests)
build: clean package

# Clean the project
clean:
	$(MVN) clean

# Package the project without running tests
package:
	$(MVN) package -DskipTests


# Combined clean and package
package-with-clean:
	$(MVN) clean package -DskipTests

# Display help
help:
	@echo "Available targets:"
	@echo "  all              - Same as 'package' (default)"
	@echo "  build            - Clean and package (without tests)"
	@echo "  clean            - Clean the project"
	@echo "  package          - Package without running tests"
	@echo "  package-with-clean - Clean and package in one command"
	@echo "  help             - Show this help message"
