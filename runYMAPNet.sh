#!/bin/bash


DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

if [[ $* == *--collab* ]]
then
 echo "Using collab mode"
elif [[ " $* " == *" --engine jax "* ]]
then
 # JAX inference needs the CUDA-enabled venv_jax (jax[cuda12]); the default
 # venv ships a CPU-only jaxlib, which falls back to CPU and then rejects the
 # cuda-exported StableHLO artifact. venv_jax is created by scripts/setup.sh.
 if [ -f venv_jax/bin/activate ]
 then
  echo "Using JAX venv (venv_jax) for GPU inference"
  source venv_jax/bin/activate
 else
  echo "ERROR: --engine jax needs venv_jax. Run scripts/setup.sh and answer 'y' to the JAX venv prompt."
  exit 1
 fi
else
 source venv/bin/activate
fi

#Use the full real-estate of the screen!
QT_AUTO_SCREEN_SCALE_FACTOR=0 QT_SCALE_FACTOR=1 python3 -m ymapnet.apps.runYMAPNet "$@"

exit 0
