#!/bin/bash
set -e

# Install furiosa compiler husk if necessary
if [ -x "$(command -v foo)" ]; then
	echo "[+] Installing furiosa-sdk pip package"
	pip install furiosa-sdk
fi

# Save results in `generated` directory (create if needed)
mkdir -p generated
cd generated

onnxes=$(find .. -name "*.onnx")
printf "[+] Target onnx files:\n$onnxes\n\n"

version=$(apt show furiosa-libcompiler | grep -E "^Version: " | cut -b 10-)
echo "[+] Compiler version from apt: $version"
revision=$(furiosa compile --version | grep "compiler" | grep -Eo 'rev: ([a-z]|[0-9])+' | cut -b 6-)
echo "[+] Compiler revision: $revision)"
mkdir -p $version\_$revision
cd $version\_$revision
for onnx in $onnxes; do
	stem=$(basename ../$onnx .onnx)
	dirn=$(dirname ../$onnx)

	# Set compiler config if exists
	if ls $dirn/*.yml 1> /dev/null 2>&1; then
		config_exists=true
		config=$(ls $dirn/*.yml)
		echo "[+] Compiler config file found for $stem: $config"
	else
		config_exists=false
		echo "[-] Compiler config file not found for $stem"
	fi

	# Single pe
	# filename=$stem\_$revision\_warboy.enf
	# if ! [ -f $filename ]; then
	# 	if [ "$config_exists" = true ]; then
	# 		( set -x; NPU_GLOBAL_CONFIG_PATH=$config furiosa compile --target-npu warboy ../$onnx -o $filename )
	# 	else
	# 		( set -x; furiosa compile --target-npu warboy ../$onnx -o $filename )
	# 	fi
	# else
	# 	echo "[+] $filename found, skipping.."
	# fi

	# Fusioned pe
	dfg_filename=$stem\_$revision\_warboy_2pe.dfg
	enf_filename=$stem\_$revision\_warboy_2pe.enf
	if [ ! -f $enf_filename ] || [ ! -f $dfg_filename ] ; then
		if [ "$config_exists" = true ]; then
			( set -x; NPU_GLOBAL_CONFIG_PATH=$config furiosa compile --target-ir dfg --target-npu warboy-2pe ../$onnx -o $dfg_filename )
			( set -x; NPU_GLOBAL_CONFIG_PATH=$config furiosa compile --target-npu warboy-2pe ../$onnx -o $enf_filename )
		else
			( set -x; furiosa compile --target-ir dfg --target-npu warboy-2pe ../$onnx -o $dfg_filename )
			( set -x; furiosa compile --target-npu warboy-2pe ../$onnx -o $enf_filename )
		fi
	else
		echo "[+] $enf_filename, $dfg_filename found, skipping.."
	fi
done
