#!/usr/bin/env python3
"""Launch a real CUDA kernel and verify device-to-host output; no ML dependencies."""
import ctypes as c
import json

cuda = c.CDLL("libcuda.so.1")


def check(name, *args):
    result = getattr(cuda, name)(*args)
    if result:
        raise RuntimeError(f"{name} failed with CUDA status {result}")


check("cuInit", 0)
device = c.c_int()
check("cuDeviceGet", c.byref(device), 0)
context = c.c_void_p()
check("cuCtxCreate_v2", c.byref(context), 0, device)
pointer = c.c_uint64()
module = c.c_void_p()
try:
    check("cuMemAlloc_v2", c.byref(pointer), c.c_size_t(4))
    ptx = b""".version 7.0
.target sm_80
.address_size 64
.visible .entry write_answer(.param .u64 output) {
 .reg .b64 %rd1;
 .reg .b32 %r1;
 ld.param.u64 %rd1, [output];
 mov.u32 %r1, 42;
 st.global.u32 [%rd1], %r1;
 ret;
}
"""
    check("cuModuleLoadData", c.byref(module), c.c_char_p(ptx))
    function = c.c_void_p()
    check("cuModuleGetFunction", c.byref(function), module, b"write_answer")
    arguments = (c.c_void_p * 1)(c.cast(c.byref(pointer), c.c_void_p))
    check("cuLaunchKernel", function, 1, 1, 1, 1, 1, 1, 0, c.c_void_p(), arguments, c.c_void_p())
    check("cuCtxSynchronize")
    answer = c.c_uint32()
    check("cuMemcpyDtoH_v2", c.byref(answer), pointer, c.c_size_t(4))
    if answer.value != 42:
        raise RuntimeError("CUDA kernel output mismatch")
    print(json.dumps({"gpu_compute_verified": True, "kernel_output": answer.value}))
finally:
    if module.value:
        check("cuModuleUnload", module)
    if pointer.value:
        check("cuMemFree_v2", pointer)
    check("cuCtxDestroy_v2", context)
