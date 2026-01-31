# Makefile

# Variables
CC = gcc
CFLAGS = -D_GNU_SOURCE -O3 -fPIC -march=native -mtune=native -pthread
LDFLAGS = -lm -ljpeg -lpng -lzstd
CODEC_DIR = codecs
SOURCE_FILES = $(CODEC_DIR)/codecs.h $(CODEC_DIR)/codecs.c $(CODEC_DIR)/asciiInput.c $(CODEC_DIR)/bmpInput.c $(CODEC_DIR)/jpgInput.c \
               $(CODEC_DIR)/pfmInput.c $(CODEC_DIR)/pngInput.c $(CODEC_DIR)/ppmInput.c $(CODEC_DIR)/pzpInput.c \
               PrepareBatch.c DataLoader.c HeatmapGenerator.c DBLoader.c cache.c
EXE_SOURCE_FILES = $(SOURCE_FILES) test.c
LIBRARY = libDataLoader.so
EXECUTABLE = test

# Check for Intel optimizations
INTEL_OPTIMIZATIONS = $(shell cat /proc/cpuinfo | grep sse3)
ifeq ($(INTEL_OPTIMIZATIONS),)
    EXTRA_FLAGS = 
else
    EXTRA_FLAGS = -DINTEL_OPTIMIZATIONS
endif

# Rules
.PHONY: all clean

all: $(LIBRARY) $(EXECUTABLE)

$(LIBRARY): $(SOURCE_FILES)
	$(CC) -shared -o $@ $(CFLAGS) $(EXTRA_FLAGS) $(SOURCE_FILES) $(LDFLAGS)

$(EXECUTABLE): $(EXE_SOURCE_FILES)
	$(CC) -o $@ -g -pg $(CFLAGS) $(EXTRA_FLAGS) $(EXE_SOURCE_FILES) $(LDFLAGS)

clean:
	rm -f $(LIBRARY) $(EXECUTABLE)

# Display GCC version
version:
	@$(CC) --version
	@echo "JIT Python/C Compilation *made by AmmarTM* handled by : "
	@$(CC) --version

# Check library dependencies
check_deps:
	@if [ -z "$(shell dpkg -l | grep build-essential)" ] || [ -z "$(shell dpkg -l | grep libjpeg-dev)" ] || [ -z "$(shell dpkg -l | grep libpng-dev)" ]; then \
		echo "Error: Unable to compile library, This probably means that you have library dependencies missing ..."; \
		echo "Try : sudo apt install build-essential libjpeg-dev libpng-dev"; \
	else \
		echo "Library Ready for use"; \
	fi

