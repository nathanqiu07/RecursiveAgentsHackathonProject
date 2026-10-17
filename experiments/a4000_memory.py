"""CUDA allocation and device-copy microbenchmark; not a model quality score."""
import ctypes as c
import json
import statistics
import time

cuda = c.CDLL('libcuda.so.1')
def check(code):
    if code:
        raise RuntimeError('CUDA error ' + str(code))
check(cuda.cuInit(0))
context = c.c_void_p()
check(cuda.cuCtxCreate_v2(c.byref(context), 0, 0))
try:
    for mib in (16, 64, 256):
        size = mib * 1024 * 1024
        src, dst = c.c_uint64(), c.c_uint64()
        check(cuda.cuMemAlloc_v2(c.byref(src), c.c_size_t(size)))
        try:
            check(cuda.cuMemAlloc_v2(c.byref(dst), c.c_size_t(size)))
            try:
                check(cuda.cuMemsetD8_v2(src, 37, c.c_size_t(size)))
                check(cuda.cuMemcpyDtoD_v2(dst, src, c.c_size_t(size)))
                check(cuda.cuCtxSynchronize())
                times = []
                for _ in range(20):
                    started = time.perf_counter()
                    check(cuda.cuMemcpyDtoD_v2(dst, src, c.c_size_t(size)))
                    check(cuda.cuCtxSynchronize())
                    times.append(time.perf_counter() - started)
                sample = (c.c_ubyte * 4096)()
                check(cuda.cuMemcpyDtoH_v2(sample, dst, c.c_size_t(len(sample))))
                assert all(x == 37 for x in sample)
                median = statistics.median(times)
                print(json.dumps({'buffer_mib': mib, 'samples': 20, 'median_ms': median * 1000, 'payload_GB_s': size / median / 1e9, 'checked_bytes': len(sample)}), flush=True)
            finally:
                check(cuda.cuMemFree_v2(dst))
        finally:
            check(cuda.cuMemFree_v2(src))
finally:
    check(cuda.cuCtxDestroy_v2(context))
