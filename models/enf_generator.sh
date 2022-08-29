#!/bin/bash
set -e

MODEL_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# Install furiosa compiler husk if necessary
if [ -x "$(command -v foo)" ]; then
	echo "[+] Installing furiosa-sdk pip package"
	pip install furiosa-sdk
fi

ONNX_FILES=($(ls $MODEL_DIR/*.onnx))
PACKAGE_VERSION=$(apt show furiosa-libcompiler 2> /dev/null | grep -E "^Version: " | cut -b 10-)
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

for INDEX in ${!ONNX_FILES[@]}; do
  ONNX_PATH=${ONNX_FILES[INDEX]}
	FULLNAME=$(basename -- "$ONNX_PATH")
  FILENAME="${FULLNAME%.*}"
  OUTPUT_PATH_BASE=${GENERATED_DIR}/${FILENAME}

  echo "[$(expr ${INDEX} + 1)/${#ONNX_FILES[@]}] Compiling $FULLNAME .."

	DFG_PATH=${OUTPUT_PATH_BASE}_warboy_2pe.dfg
	ENF_PATH=${OUTPUT_PATH_BASE}_warboy_2pe.enf

	# Set compiler config if exists
	unset NPU_COMPILER_CONFIG_PATH
	if [ -f $MODEL_DIR/${FILENAME}.yaml ]; then
		export NPU_COMPILER_CONFIG_PATH=$MODEL_DIR/${FILENAME}.yaml
	fi

  printf " [Task 1/2] Generating $(basename -- $DFG_PATH)"
	if [ -f $DFG_PATH ]; then
    echo " ... (Skipped)"
  else
    echo " ... (Running)"
    if [ ! -z $NPU_COMPILER_CONFIG_PATH ];then
      echo "    Using $(basename -- $NPU_COMPILER_CONFIG_PATH)"
    fi
    furiosa compile --target-ir dfg --target-npu warboy-2pe $ONNX_PATH -o $DFG_PATH &> /dev/null
  fi

  printf " [Task 2/2] Generating $(basename -- $ENF_PATH)"
	if [ -f $ENF_PATH ]; then
    echo " ... (Skipped)"
  else
    echo " ... (Running)"
    if [ ! -z $NPU_COMPILER_CONFIG_PATH ];then
      echo "    Using $(basename -- $NPU_COMPILER_CONFIG_PATH)"
    fi
    furiosa compile --target-ir enf --target-npu warboy-2pe $ONNX_PATH -o $ENF_PATH &> /dev/null
  fi
done
