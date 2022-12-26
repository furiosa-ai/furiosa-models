
from ctypes import CFUNCTYPE, POINTER, c_float, c_uint32

from llvmlite import binding as llvm
from llvmlite import ir as lc
import numpy as np

llvm.initialize()
llvm.initialize_native_target()
llvm.initialize_native_asmprinter()

print("parse assembler.")
with open("box_decode.ll", "r") as f:
    m = f.read()
modref = llvm.parse_assembly(m)

u32 = c_uint32
f32 = c_float
u32p = np.ctypeslib.ndpointer(dtype=u32, ndim=1, flags='C_CONTIGUOUS')
f32p = np.ctypeslib.ndpointer(dtype=f32, ndim=1, flags='C_CONTIGUOUS')

def create_execution_engine():
    target = llvm.Target.from_default_triple()
    target_machine = target.create_target_machine()
    backing_mod = llvm.parse_assembly("")
    engine = llvm.create_mcjit_compiler(backing_mod, target_machine)
    return engine
llvm_engine = create_execution_engine()
llvm_engine.add_module(modref)
llvm_engine.finalize_object()


func_ptr = llvm_engine.get_function_address("box_decode_feat")
function = CFUNCTYPE( None, 
                     f32p,  # anchors
                     u32, # num_anchors
                     f32,  # stride
                     f32,  # conf_thres
                     u32,  # max_boxes
                     f32p,   # feat
                     u32,  # batchsize
                     u32,  # ny
                     u32,  # nx   
                     u32,  # no
                     f32p,   # out_batch
                     u32p       # out_batch_pos
                     )(func_ptr)

anchors = np.zeros([5,30], dtype=np.float32)
num_class = 8
num_output_per_ancher = num_class + 5
num_anchors = 3
stride = 0.3
conf_thresh = 0.02
max_boxes = 30
width=20
height=20
feat = np.zeros([1,num_anchors,height,width,num_output_per_ancher], dtype=np.float32)
batch_size, na, ny, nx, no = feat.shape
out_batch = np.empty( (batch_size, max_boxes, 6), dtype=np.float32)
out_batch_pos = np.zeros(batch_size, dtype=np.uint32)


print("function1", function( anchors.reshape(-1),
                            int(num_anchors),
                            stride, 
                            conf_thresh, 
                            max_boxes,
                            feat.reshape(-1),
                            batch_size,
                            ny,
                            nx,
                            no,
                            out_batch.reshape(-1),
                            out_batch_pos))

llvm.shutdown()