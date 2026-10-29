# Makefile

# Variables
CC = gcc
CFLAGS = -D_GNU_SOURCE -O3 -fPIC -march=native -mtune=native -pthread -DPZPDIR_WITH_PZP=0
LDFLAGS = -lm -ljpeg -lpng -lzstd -llz4
CODEC_DIR = codecs
SOURCE_FILES = $(CODEC_DIR)/codecs.h $(CODEC_DIR)/codecs.c $(CODEC_DIR)/asciiInput.c $(CODEC_DIR)/bmpInput.c $(CODEC_DIR)/jpgInput.c \
               $(CODEC_DIR)/pfmInput.c $(CODEC_DIR)/pngInput.c $(CODEC_DIR)/ppmInput.c $(CODEC_DIR)/pzpInput.c \
               PrepareBatch.c DataLoader.c HeatmapGenerator.c DBLoader.c cache.c PZPDLoader.c
EXE_SOURCE_FILES = $(SOURCE_FILES) test.c
LIBRARY = libDataLoader.so
# pzpdir (.pzpd archives), vendored in pzpdir/, is built as its own library and linked with DT_RPATH
# $ORIGIN/pzpdir (found next to libDataLoader.so / the executables before anything on LD_LIBRARY_PATH)
PZPDIR_LIB = pzpdir/libpzpdir.so
PZPDIR_SRC = $(wildcard pzpdir/pzpdir_*.c)
PZPDIR_LINK = -Lpzpdir -lpzpdir -Wl,-rpath,'$$ORIGIN/pzpdir' -Wl,--disable-new-dtags
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

$(PZPDIR_LIB): $(PZPDIR_SRC) $(wildcard pzpdir/*.h)
	$(CC) -shared -o $@ $(CFLAGS) $(EXTRA_FLAGS) $(PZPDIR_SRC) -lzstd -llz4

$(LIBRARY): $(SOURCE_FILES) $(PZPDIR_LIB)
	$(CC) -shared -o $@ $(CFLAGS) $(EXTRA_FLAGS) $(SOURCE_FILES) $(PZPDIR_LINK) $(LDFLAGS)

$(EXECUTABLE): $(EXE_SOURCE_FILES) $(PZPDIR_LIB)
	$(CC) -o $@ -g -pg $(CFLAGS) $(EXTRA_FLAGS) $(EXE_SOURCE_FILES) $(PZPDIR_LINK) $(LDFLAGS)

clean:
	rm -f $(LIBRARY) $(EXECUTABLE) $(PZPDIR_LIB)

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

