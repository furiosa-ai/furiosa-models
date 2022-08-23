# Dfg & enf generator

## See all available compiler versions
Running the following command will show the available compiler versions except the nightly ones.
```bash
apt-cache madison furiosa-libcompiler | grep -v "+nightly" | tac | awk -F"|" '{print $2;}' | awk '{$1=$1};1
```

## Run dfg & enf generator
After installing the appropriate `furiosa-libcompiler`, running the script will generate dfg and enf for warboy-2pe for each onnx files under the current directory.
Outputs will be saved under `generated` directory.
```bash
./enf_generator.sh
```
