#!/bin/bash

# Reset
Color_Off='\033[0m'       # Text Reset

# Regular Colors
Black='\033[0;30m'        # Black
Red='\033[0;31m'          # Red
Green='\033[0;32m'        # Green
Yellow='\033[0;33m'       # Yellow
Blue='\033[0;34m'         # Blue
Purple='\033[0;35m'       # Purple
Cyan='\033[0;36m'         # Cyan
White='\033[0;37m'        # White

# Bold
BBlack='\033[1;30m'       # Black
BRed='\033[1;31m'         # Red
BGreen='\033[1;32m'       # Green
BYellow='\033[1;33m'      # Yellow
BBlue='\033[1;34m'        # Blue
BPurple='\033[1;35m'      # Purple
BCyan='\033[1;36m'        # Cyan
BWhite='\033[1;37m'       # White


DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

echo -e "$Cyan JIT Python/C Compilation *made by AmmarTM* handled by : $Color_Off"

versionToBuild=`cat DataLoader.h | grep "const char version" | cut -d\" -f2`
echo -e "$Cyan Building $versionToBuild Dataloader $Color_Off"
gcc --version


OPENMP="-fopenmp" #Sobel 23.39 with OpenMP
OPENMP=" " #OpenMP disabled

CODEC_DIR="codecs/"

SOURCE="
$CODEC_DIR/codecs.c
$CODEC_DIR/asciiInput.c
$CODEC_DIR/bmpInput.c
$CODEC_DIR/jpgInput.c
$CODEC_DIR/pfmInput.c
$CODEC_DIR/pngInput.c
$CODEC_DIR/ppmInput.c
$CODEC_DIR/pzpInput.c
PrepareBatch.c
DataLoader.c
HeatmapGenerator.c
DBLoader.c
cache.c
"

INTEL_OPTIMIZATIONS=`cat /proc/cpuinfo | grep sse3`
if [ -z "$INTEL_OPTIMIZATIONS" ] ; then
 echo "No intel optimizations available"
 EXTRA_FLAGS=" "
else
 echo "Intel Optimizations available and will be used"
 EXTRA_FLAGS="-DINTEL_OPTIMIZATIONS -mavx2"
fi
 
#Make sure previous files are wiped
rm ./libDataLoader.so ./test

#This is the library to be linked to Python code
gcc -shared -o libDataLoader.so $EXTRA_FLAGS $OPENMP -D_GNU_SOURCE -O3 -fPIC -Wno-unused-function -march=native -mtune=native $SOURCE -pthread -lm -lpng -ljpeg -lzstd

#This executable is for optimized run
gcc -o opt_test $EXTRA_FLAGS $OPENMP -D_GNU_SOURCE -O3 -fPIE -fPIC -march=native -mtune=native $SOURCE test.c -pthread -lm -lpng -ljpeg -lzstd

#This executable is for debugging
#https://cheatsheetseries.owasp.org/cheatsheets/C-Based_Toolchain_Hardening_Cheat_Sheet.html     -fsanitize=thread
gcc -o test $EXTRA_FLAGS $OPENMP -D_GNU_SOURCE -O0 -g3 -fno-omit-frame-pointer -pg -Wstrict-overflow -fsanitize=address -fPIE -fPIC -Wno-unused-function -march=native -mtune=native $SOURCE test.c -pthread -lm -lpng -ljpeg -lzstd

#This executable is for profiling
gcc -o profile_test $EXTRA_FLAGS $OPENMP -D_GNU_SOURCE -O0 -g3 -fno-omit-frame-pointer -pg -Wstrict-overflow -fPIE -fPIC -Wno-unused-function -march=native -mtune=native $SOURCE test.c -pthread -lm -lpng -ljpeg -lzstd


gcc -o pzp codecs/pzp.c -pthread -lm -lpng -ljpeg -lzstd

if [ $? -ne 0 ]; then
    echo -e "$BRed Error: Unable to compile library $Color_Off"
    echo "This probably means that you have library dependencies missing ..."
    echo "Try : sudo apt install build-essential libturbojpeg-dev libpng-dev" # libjpeg-dev 
else
    echo -e "$BGreen C DataLoader Library Ready for use $Color_Off"
fi

#Note to verify if libjpeg-turbo is used (4x speed) issue : 
#dpkg -S /usr/lib/x86_64-linux-gnu/libjpeg.so.8
#it should return libjpeg-turbo8:am64 

exit 0
