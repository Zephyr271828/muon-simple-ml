#!/bin/bash
set +x

CMD="$1"
shift || true

if [ -z "$CMD" ]; then
  # 
  echo "[INFO] No command provided → entering interactive Apptainer shell..."
  apptainer exec \
    --nv \
    --bind /work/hdd/bdhh:/work/hdd/bdhh \
    --bind /work/nvme/bdhh:/work/nvme/bdhh \
    --bind /projects/bdhh \
    --env PATH=/vllm-workspace/.venv/bin:$PATH \
    /work/nvme/bdhh/yxu21/containers/vllm-gh200-openai.sif \
    /bin/bash -i -c "
    export PATH=/work/nvme/bdhh/yxu21/venvs/synlogic/bin/python:/vllm-workspace/.venv/bin:\$PATH
    exec bash -i
    "
  exit 0
fi

if [ -f "$CMD" ]; then
  if [[ "$CMD" == *.sh ]]; then
    CMD="bash $CMD $*"
  elif [[ "$CMD" == *.py ]]; then
    CMD="python $CMD $*"
  fi
else
  CMD="$CMD $*"
fi
echo 


COMMANDS="
source /work/nvme/bdhh/yxu21/venvs/synlogic/bin/activate

unset SSL_CERT_FILE
unset SSL_CERT_DIR
unset REQUESTS_CA_BUNDLE
unset CURL_CA_BUNDLE
unset HTTPS_PROXY
unset HTTP_PROXY
unset ALL_PROXY
unset ROCR_VISIBLE_DEVICES

TOKENIZERS_PARALLELISM=true
VLLM_DISABLE_USAGE_STATS=1
VLLM_NO_USAGE_STATS=1
VLLM_USAGE_STATS=0

${CMD}
"

apptainer exec \
  --nv \
  --bind /work/nvme/bdhh:/work/nvme/bdhh \
  --bind /work/hdd/bdhh:/work/hdd/bdhh \
  --bind /projects/bdhh:/projects/bdhh \
  --env PATH=/vllm-workspace/.venv/bin:$PATH \
  /work/nvme/bdhh/yxu21/containers/vllm-gh200-openai.sif \
  /bin/bash -lc "$COMMANDS"