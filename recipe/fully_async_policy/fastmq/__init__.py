"""
FastMQ - Ultra-low-latency shared-memory SPSC (Single Producer Single Consumer) ring buffer

This module provides lock-free, shared-memory based message queue primitives
implemented in C++ with pybind11 bindings for Python.

Classes:
    SPSC: Single Producer Single Consumer queue using POSIX shared memory
    
Example:
    >>> from recipe.fully_async_policy import fastmq
    >>> # Producer side (create=True)
    >>> producer_q = fastmq.SPSC("/my_queue", capacity_bytes=1024*1024, create=True)
    >>> producer_q.try_push(b"hello world")
    >>> 
    >>> # Consumer side (create=False, opens existing)
    >>> consumer_q = fastmq.SPSC("/my_queue", capacity_bytes=0, create=False)
    >>> data = consumer_q.try_pop()
    >>> 
    >>> # Cleanup (call from creator)
    >>> fastmq.SPSC.unlink("/my_queue")
"""

try:
    # Import the compiled C++ extension module
    # The .so file is named fastmq.cpython-312-x86_64-linux-gnu.so
    # Python will automatically find it when we import .fastmq
    from .fastmq import SPSC
    
    # Export the SPSC class
    __all__ = ['SPSC']
    
    # Expose commonly used functionality
    # Note: SPSC.unlink is a static method accessible via SPSC.unlink(name)
    
except ImportError as e:
    import warnings
    warnings.warn(
        f"Failed to import fastmq C++ extension: {e}\n"
        "The fastmq module may not be compiled. "
        "Please compile fastmq.cpp or use the fallback message queue implementation.",
        ImportWarning
    )
    # Provide a stub to prevent immediate crashes
    SPSC = None
    __all__ = []
