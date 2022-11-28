#!/bin/bash
set -e

MODEL_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

ONNX_FILES=($(ls $MODEL_DIR/*.onnx))
PACKAGE_VERSION=$(apt list furiosa-libcompiler -a 2> /dev/null | grep installed | awk '{print $2}')
COMPILER_VERSION=$(furiosa compile --version | grep "compiler" | awk '{print $2}')
COMPILER_REVISION=$(furiosa compile --version | grep "compiler" | grep -Eo 'rev: ([a-z]|[0-9])+' | cut -b 6-)
COMPILER_FULL_VERSION=${COMPILER_VERSION}_${COMPILER_REVISION}

echo "[+] Detected version of compiler: $COMPILER_VERSION (rev. $COMPILER_REVISION)"
echo "[+] Installed version of compiler package: $PACKAGE_VERSION"
echo "[+] Found ${#ONNX_FILES[@]} ONNX Files:"
for INDEX in ${!ONNX_FILES[@]};do
  echo " [$(expr ${INDEX} + 1)] "$(basename -- ${ONNX_FILES[INDEX]})
done

GENERATED_DIR=${MODEL_DIR}/generated/${COMPILER_FULL_VERSION}
mkdir -p ${GENERATED_DIR}
echo "[+] Output directory: ${GENERATED_DIR}"

function compile() {
  OUTPUT_PATH=$1
  FORMAT=$2

  if [ -f $OUTPUT_PATH ]; then
    echo " ... (Skipped)"
  else
    echo " ... (Running)"

    COMPILE_CMD="furiosa compile --target-npu warboy-2pe --target-ir ${FORMAT} $ONNX_PATH -o ${OUTPUT_PATH}"

    # Try to find the format-specific compiler config
    IR_COMPILER_CONFIG=$MODEL_DIR/${FILENAME}.${FORMAT}.yaml
    if [ -f $IR_COMPILER_CONFIG ]; then
      echo "    >> Overridden by $(basename -- $IR_COMPILER_CONFIG)"
      env NPU_COMPILER_CONFIG_PATH=$IR_COMPILER_CONFIG bash -c "$COMPILE_CMD" &> /dev/null
    else
      if [ ! -z $NPU_COMPILER_CONFIG_PATH ]; then
        echo "    >> Using $(basename -- $NPU_COMPILER_CONFIG_PATH)"
      fi
      $COMPILE_CMD &> /dev/null
    fi
  fi
}

for INDEX in ${!ONNX_FILES[@]}; do
  ONNX_PATH=${ONNX_FILES[INDEX]}
	FULLNAME=$(basename -- "$ONNX_PATH")
  FILENAME="${FULLNAME%.*}"
  OUTPUT_PATH_BASE=${GENERATED_DIR}/${FILENAME}

  echo "[$(expr ${INDEX} + 1)/${#ONNX_FILES[@]}] Compiling $FULLNAME .."

  # Try to find the compiler config for all IR formats
  unset NPU_COMPILER_CONFIG_PATH
  if [ -f $MODEL_DIR/${FILENAME}.yaml ]; then
    export NPU_COMPILER_CONFIG_PATH=$MODEL_DIR/${FILENAME}.yaml
    echo "  > Compiler config found at $(basename -- $NPU_COMPILER_CONFIG_PATH)"
  fi

	DFG_PATH=${OUTPUT_PATH_BASE}_warboy_2pe.dfg
	ENF_PATH=${OUTPUT_PATH_BASE}_warboy_2pe.enf

  printf " [Task 1/2] Generating $(basename -- $DFG_PATH)"
	compile $DFG_PATH dfg

  printf " [Task 2/2] Generating $(basename -- $ENF_PATH)"
  compile $ENF_PATH enf
done
