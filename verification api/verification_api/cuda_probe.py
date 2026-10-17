"""Isolated, standard-library-only CUDA compute smoke test for node enrollment."""

import ctypes as c
import json

PTX = b"""
.version 6.0
.target sm_50
.address_size 64
.visible .entry verify_node(.param .u64 output) {
    .reg .b64 %rd1;
    .reg .b32 %r1;
    ld.param.u64 %rd1, [output];
    mov.u32 %r1, 42;
    st.global.u32 [%rd1], %r1;
    ret;
}
"""


def probe():
    driver = c.CDLL("libcuda.so.1")
    signatures = {
        "cuInit": [c.c_uint],
        "cuDriverGetVersion": [c.POINTER(c.c_int)],
        "cuDeviceGetCount": [c.POINTER(c.c_int)],
        "cuDeviceGet": [c.POINTER(c.c_int), c.c_int],
        "cuDeviceGetName": [c.c_char_p, c.c_int, c.c_int],
        "cuDeviceTotalMem_v2": [c.POINTER(c.c_size_t), c.c_int],
        "cuDeviceComputeCapability": [c.POINTER(c.c_int), c.POINTER(c.c_int), c.c_int],
        "cuCtxCreate_v2": [c.POINTER(c.c_void_p), c.c_uint, c.c_int],
        "cuCtxDestroy_v2": [c.c_void_p],
        "cuModuleLoadData": [c.POINTER(c.c_void_p), c.c_void_p],
        "cuModuleUnload": [c.c_void_p],
        "cuModuleGetFunction": [c.POINTER(c.c_void_p), c.c_void_p, c.c_char_p],
        "cuMemAlloc_v2": [c.POINTER(c.c_uint64), c.c_size_t],
        "cuMemFree_v2": [c.c_uint64],
        "cuMemcpyDtoH_v2": [c.c_void_p, c.c_uint64, c.c_size_t],
        "cuCtxSynchronize": [],
        "cuLaunchKernel": [
            c.c_void_p,
            *([c.c_uint] * 7),
            c.c_void_p,
            c.POINTER(c.c_void_p),
            c.POINTER(c.c_void_p),
        ],
    }
    for name, args in signatures.items():
        function = getattr(driver, name)
        function.argtypes = args
        function.restype = c.c_int

    def call(name, *args):
        result = getattr(driver, name)(*args)
        if result:
            raise RuntimeError(f"{name} failed (CUDA error {result})")

    call("cuInit", 0)
    count, version = c.c_int(), c.c_int()
    call("cuDeviceGetCount", c.byref(count))
    call("cuDriverGetVersion", c.byref(version))
    if count.value < 1:
        raise RuntimeError("No accessible CUDA devices")
    devices = []
    for ordinal in range(count.value):
        device, major, minor = c.c_int(), c.c_int(), c.c_int()
        name, memory = c.create_string_buffer(256), c.c_size_t()
        call("cuDeviceGet", c.byref(device), ordinal)
        call("cuDeviceGetName", name, len(name), device)
        call("cuDeviceTotalMem_v2", c.byref(memory), device)
        call("cuDeviceComputeCapability", c.byref(major), c.byref(minor), device)
        context, module, function = c.c_void_p(), c.c_void_p(), c.c_void_p()
        allocation = c.c_uint64()
        call("cuCtxCreate_v2", c.byref(context), 0, device)
        try:
            ptx = c.create_string_buffer(PTX)
            call("cuModuleLoadData", c.byref(module), c.cast(ptx, c.c_void_p))
            call("cuModuleGetFunction", c.byref(function), module, b"verify_node")
            call("cuMemAlloc_v2", c.byref(allocation), 4)
            parameters = (c.c_void_p * 1)(c.cast(c.byref(allocation), c.c_void_p))
            call("cuLaunchKernel", function, 1, 1, 1, 1, 1, 1, 0, None, parameters, None)
            call("cuCtxSynchronize")
            value = c.c_uint32()
            call("cuMemcpyDtoH_v2", c.byref(value), allocation, 4)
            if value.value != 42:
                raise RuntimeError("CUDA compute result did not match the expected value")
            devices.append(
                {
                    "ordinal": ordinal,
                    "name": name.value.decode(),
                    "memory_bytes": memory.value,
                    "compute_capability": f"{major.value}.{minor.value}",
                    "compute_test": "passed",
                }
            )
        finally:
            if allocation.value:
                driver.cuMemFree_v2(allocation)
            if module.value:
                driver.cuModuleUnload(module)
            driver.cuCtxDestroy_v2(context)
    return {"driver_cuda_version": version.value, "gpus": devices}


if __name__ == "__main__":
    try:
        print(json.dumps(probe()))
    except (OSError, AttributeError, RuntimeError) as exc:
        print(json.dumps({"error": str(exc)}))
        raise SystemExit(1)
