#!/bin/bash
THISDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$THISDIR"

rm callgrind.out.* 

./makeLibrary.sh
strace -T -tt -o strace_output.log -e trace=open,read,write,close ./test $@
awk '{print $1 "," $2 "," $NF}' strace_output.log > strace_data.csv

kcachegrind
exit 0
