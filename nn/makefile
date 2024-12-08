# #
# # This makefile does many things by convention which means
# # less fiddling around with the makefile. It is assumed that
# # tests end with an _test.c or _test.cpp suffix. All other source
# # files are compiled to a single executable, whose main name
# # is given with the TARGET_NAME macro.
# #
# # Written by Sebastian Bauer <mail@sebastianbauer.info> for
# # lectures at the HTW.
# #
# #	First Expanded by Thorne Wiggers
# #	Later expanded by Roman Roters-Coca
# #
# #	Special Thanks to Thorne for sharing this makefile.
# #
# # This makefile is released in public domain.

# Default configuration
TARGET_NAME=main
BUILD_DIR=build

# Flags for gcc
OPTIMIZE=-O3
DEBUG=1
CFLAGS= $(OPTIMIZE) -Wall -MMD -MP -fopenmp -mavx -lm
LDFLAGS=-fopenmp -mavx
LDLIBS=-lm

ifeq ($(DEBUG), 1)
CFLAGS += -g -ggdb
endif

# Compiler
CC=gcc

# Locate source files
SRCS=$(filter-out %_test.c,$(wildcard *.c))

# Separate object files for each version
SEQ_OBJS=$(addprefix $(BUILD_DIR)/seq_,$(patsubst %.c,%.o,$(SRCS)))
PAR_OBJS=$(addprefix $(BUILD_DIR)/parallel_,$(patsubst %.c,%.o,$(SRCS)))
SIMD_OBJS=$(addprefix $(BUILD_DIR)/simd_,$(patsubst %.c,%.o,$(SRCS)))

# Executables
SEQ_EXEC=$(BUILD_DIR)/$(TARGET_NAME)_seq
PAR_EXEC=$(BUILD_DIR)/$(TARGET_NAME)_parallel
SIMD_EXEC=$(BUILD_DIR)/$(TARGET_NAME)_simd

# Default target
.PHONY: all
all: compile-seq compile-parallel compile-simd test

# Compile sequential version
.PHONY: compile-seq
compile-seq: CFLAGS += -DSEQ
compile-seq: $(SEQ_EXEC)

$(SEQ_EXEC): $(SEQ_OBJS)
	$(CC) $(LDFLAGS) $(SEQ_OBJS) $(LDLIBS) -o $@

$(BUILD_DIR)/seq_%.o: %.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

# Compile parallel version
.PHONY: compile-parallel
compile-parallel: CFLAGS += -DPARALLEL
compile-parallel: $(PAR_EXEC)

$(PAR_EXEC): $(PAR_OBJS)
	$(CC) $(LDFLAGS) $(PAR_OBJS) $(LDLIBS) -o $@

$(BUILD_DIR)/parallel_%.o: %.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

# Compile SIMD version
.PHONY: compile-simd
compile-simd: CFLAGS += -DSIMD
compile-simd: $(SIMD_EXEC)

$(SIMD_EXEC): $(SIMD_OBJS)
	$(CC) $(LDFLAGS) $(SIMD_OBJS) $(LDLIBS) -o $@

$(BUILD_DIR)/simd_%.o: %.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

# Compile all versions
.PHONY: compile-all
compile-all: compile-seq compile-parallel compile-simd

# Build directory
$(BUILD_DIR):
	@mkdir -p $(BUILD_DIR)

# Test target
.PHONY: test
test:
	$(foreach T,$(TESTS), ./$T &&) true

# Clean target
.PHONY: clean
clean:
	rm -Rf $(BUILD_DIR)

# Run sequential version with an optional config file or command-line arguments
.PHONY: run-seq
run-seq:
	@if [ ! -x $(SEQ_EXEC) ]; then \
		echo "Error: $(SEQ_EXEC) does not exist. Compile it first with 'make compile-seq'"; \
		exit 1; \
	fi; \
	if [ "$(CONFIG)" != "" ]; then \
		echo "Running sequential program with config file: $(CONFIG)"; \
		./$(SEQ_EXEC) "$(CONFIG)"; \
	elif [ "$(ARGS)" != "" ]; then \
		echo "Running sequential program with arguments: $(ARGS)"; \
		./$(SEQ_EXEC) $(ARGS); \
	else \
		echo "Running sequential program with default configuration"; \
		./$(SEQ_EXEC); \
	fi

# Run parallel version with an optional config file or command-line arguments
.PHONY: run-parallel
run-parallel:
	@if [ ! -x $(PAR_EXEC) ]; then \
		echo "Error: $(PAR_EXEC) does not exist. Compile it first with 'make compile-parallel'"; \
		exit 1; \
	fi; \
	if [ "$(CONFIG)" != "" ]; then \
		echo "Running parallel program with config file: $(CONFIG)"; \
		./$(PAR_EXEC) "$(CONFIG)"; \
	elif [ "$(ARGS)" != "" ]; then \
		echo "Running parallel program with arguments: $(ARGS)"; \
		./$(PAR_EXEC) $(ARGS); \
	else \
		echo "Running parallel program with default configuration"; \
		./$(PAR_EXEC); \
	fi

# Run SIMD version with an optional config file or command-line arguments
.PHONY: run-simd
run-simd:
	@if [ ! -x $(SIMD_EXEC) ]; then \
		echo "Error: $(SIMD_EXEC) does not exist. Compile it first with 'make compile-simd'"; \
		exit 1; \
	fi; \
	if [ "$(CONFIG)" != "" ]; then \
		echo "Running SIMD program with config file: $(CONFIG)"; \
		./$(SIMD_EXEC) "$(CONFIG)"; \
	elif [ "$(ARGS)" != "" ]; then \
		echo "Running SIMD program with arguments: $(ARGS)"; \
		./$(SIMD_EXEC) $(ARGS); \
	else \
		echo "Running SIMD program with default configuration"; \
		./$(SIMD_EXEC); \
	fi

# Help target
.PHONY: help
help:
	@echo "Available targets:"
	@echo "  all                - Compile all versions (sequential, parallel, SIMD) and run unit tests"
	@echo "  compile-seq        - Compile the sequential version"
	@echo "  compile-parallel   - Compile the parallel version (with omp library)"
	@echo "  compile-simd       - Compile the SIMD version"
	@echo "  compile-all        - Compile all versions"
	@echo "  test               - Run tests"
	@echo "  clean              - Clean the build directory"
	@echo "  run-seq            - Run the sequential version"
	@echo "  run-parallel       - Run the parallel version (with omp library)"
	@echo "  run-simd           - Run the SIMD version"
	@echo "  docs               - Generate documentation using Doxygen"

# Docs target
.PHONY: docs
docs:
	@mkdir -p docs
	@doxygen DOXYFILE

-include $(DEPS)
