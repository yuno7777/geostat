"""Process memory measurements for the standalone benchmarks."""

import os
import sys
from collections.abc import Callable
from functools import lru_cache


@lru_cache(maxsize=1)
def _windows_memory_reader() -> Callable[[], tuple[int, int]]:
    import ctypes
    from ctypes import wintypes

    class ProcessMemoryCounters(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    load_dll = vars(ctypes)["WinDLL"]
    current_process = load_dll("kernel32", use_last_error=True).GetCurrentProcess
    current_process.restype = wintypes.HANDLE
    get_memory = load_dll("psapi", use_last_error=True).GetProcessMemoryInfo
    get_memory.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(ProcessMemoryCounters),
        wintypes.DWORD,
    ]
    get_memory.restype = wintypes.BOOL

    def read() -> tuple[int, int]:
        counters = ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        if not get_memory(current_process(), ctypes.byref(counters), counters.cb):
            raise OSError("Could not read the process peak working set")
        return int(counters.WorkingSetSize), int(counters.PeakWorkingSetSize)

    return read


def peak_mib() -> float:
    if sys.platform == "win32":
        return _windows_memory_reader()()[1] / 1024**2

    import resource

    # Linux reports KiB and macOS reports bytes.
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (
        1024**2 if sys.platform == "darwin" else 1024
    )


def resident_bytes() -> int:
    if sys.platform == "win32":
        return _windows_memory_reader()()[0]
    if sys.platform == "linux":
        with open("/proc/self/statm", encoding="ascii") as stat:
            resident_pages = int(stat.read().split()[1])
        return resident_pages * os.sysconf("SC_PAGE_SIZE")
    raise OSError("Current process memory sampling is supported on Windows and Linux")
