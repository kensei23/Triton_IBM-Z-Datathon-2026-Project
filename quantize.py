import ctypes

import llama_cpp
from config import settings

SRC = f"{settings['model_dir']}/{settings['model_file']}"
DST = f"{settings['model_dir']}/{settings['quantized_file']}"

# starts from the default params, then just change the target type
params = llama_cpp.llama_model_quantize_default_params()
params.ftype = llama_cpp.LLAMA_FTYPE_MOSTLY_Q8_0
params.nthread = settings["n_threads"]

llama_cpp.llama_backend_init()
ret = llama_cpp.llama_model_quantize(SRC.encode(), DST.encode(), ctypes.byref(params))
print("quantize returned", ret, "(0 means ok)")
