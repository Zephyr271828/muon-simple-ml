#!/bin/bash
CMD="$1"

if [ -z "$CMD" ]; then
  echo "❌ Error: No command provided."
  echo "Usage:"
  echo "  ./launch.sh \"<command>\" or ./launch.sh <script_path>"
  exit 1
fi

if [ -f "$CMD" ]; then
  # CMD="bash $CMD"

  if [[ "$CMD" == *.sh ]]; then
    JOB_NAME=$(basename "$CMD" .sh) 
    CMD="$(sed -e '/^[[:space:]]*#/d' \
      -e 's/"/\\"/g' \
      -e 's/\$/\\$/g' "$CMD")"
  elif [[ "$CMD" == *.py ]]; then
    JOB_NAME=$(basename "$CMD" .py)
    CMD="python $CMD"
  fi
else
  CMD="$CMD"
  JOB_NAME='cool_job'
fi

for arg in "$@"; do
  case $arg in
    --job_name=*)
      JOB_NAME="${arg#*=}"
      ;;
    --time=*)
      TIME="${arg#*=}"
      ;;
    --gpus=*)
      GPUS="${arg#*=}"
      ;;
    --nodes=*)
      NODES="${arg#*=}"
      ;;
    --mem=*)
      MEM="${arg#*=}"
      ;;
    --account=*)
      ACCOUNT="${arg#*=}"
      ;;
    --tag=*)
      TAG="${arg#*=}"
      ;;
    *)
      SBATCH_FLAGS="$SBATCH_FLAGS\n#SBATCH $arg"
      ;;
  esac
done

TIME=${TIME:-"1:00:00"}
GPUS=${GPUS:-1}
NODES=${NODES:-1}
MEM=${MEM:-"512GB"}
ACCOUNT=${ACCOUNT:-"bdhh"}
# TAG=#${TAG:-""}

# JOB_NAME=${2:-$JOB_NAME}
# DRY_RUN=$3

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

${CMD}
"

# truncate job name
if [ ${#JOB_NAME} -gt 200 ]; then
  JOB_NAME="${JOB_NAME:0:200}"
fi


JOB_SCRIPT=$(mktemp tmp/launch_job_XXXXXX.sh)
LOG_OUT="logs/${JOB_NAME}.out"
LOG_ERR="logs/${JOB_NAME}.err"

cat > "$JOB_SCRIPT" <<EOF
#!/bin/bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --output=$LOG_OUT
#SBATCH --error=$LOG_ERR
#SBATCH --account=$ACCOUNT-dtai-gh
#SBATCH --partition=ghx4

#SBATCH --nodes=${NODES}
#SBATCH --ntasks-per-node=1

#SBATCH --cpus-per-task=72
#SBATCH --mem=${MEM}
#SBATCH --time=${TIME}
#SBATCH --gres=gpu:${GPUS}

#SBATCH --mail-type=all
#SBATCH --mail-user=yufengx@illinois.edu
#SBATCH --requeue

apptainer exec \\
  --nv \\
  --bind /work/nvme/bdhh/yxu21:/work/nvme/bdhh/yxu21 \\
  --bind /work/hdd/bdhh:/work/hdd/bdhh \\
  --bind /projects/bdhh/yxu21:/projects/bdhh/yxu21 \\
  --env TOKENIZERS_PARALLELISM=true \\
  --env VLLM_DISABLE_USAGE_STATS=1 \\
  --env VLLM_NO_USAGE_STATS=1 \\
  --env VLLM_USAGE_STATS=0 \\
  /work/nvme/bdhh/yxu21/containers/vllm-gh200-openai.sif \\
  /bin/bash -lc "export SLURM_JOB_NUM_NODES=${NODES} && $COMMANDS"
EOF

echo $COMMANDS

if [ -z "$DRY_RUN" ]; then

  # if [[ -f "$LOG_OUT" || -f "$LOG_ERR" ]]; then
  #   echo "⚠️  Log files already exist:"
  #   [[ -f "$LOG_OUT" ]] && echo " - $(realpath "$LOG_OUT")"
  #   [[ -f "$LOG_ERR" ]] && echo " - $(realpath "$LOG_ERR")"
  #   read -p "Do you want to overwrite them? [y/N]: " confirm
  #   confirm=${confirm,,}  # lowercase
  # fi
  confirm=${confirm:-"y"}  # default to 'y' if empty
  if [[ "$confirm" == "y" || "$confirm" == "yes" ]]; then
    sbatch_output=$(sbatch "$JOB_SCRIPT")
    job_id=$(echo "$sbatch_output" | awk '{print $4}')
    echo "✅ Submitted job with script $JOB_SCRIPT"
    echo "See logs at $LOG_OUT and $LOG_ERR"
  else
    echo "❌ Aborted by user."
  fi
else
  echo "dry run mode. See sbatch script at $JOB_SCRIPT"
fi

#SBATCH --exclude=gh060,gh091,gh009,gh055,gh142,gh035