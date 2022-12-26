#!/bin/bash

#clang -emit-llvm -S box_decode.cpp -o box_decode.ll
python create-llvm-ir-from-source-file.py box_decode.cpp . clang ./