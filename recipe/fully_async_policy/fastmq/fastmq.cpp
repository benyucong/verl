// g++ -O3 -std=c++17 -fPIC -shared fastmq.cpp -o fastmq$(python3 -c "import sysconfig;print(sysconfig.get_config_var('EXT_SUFFIX'))") -I$(python3 -c "import pybind11,sys;print(pybind11.get_include())")

#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <atomic>
#include <cstring>
#include <stdexcept>
#include <string>

namespace py = pybind11;

static inline size_t next_pow2(size_t x) {
    if (x < 2) return 2;
    --x; x |= x>>1; x |= x>>2; x |= x>>4; x |= x>>8; x |= x>>16; x |= x>>32; return x+1;
}

struct alignas(64) Header {
    std::atomic<size_t> head;  // consumer reads from head
    char pad1[64 - sizeof(std::atomic<size_t>)];
    std::atomic<size_t> tail;  // producer writes at tail
    char pad2[64 - sizeof(std::atomic<size_t>)];
    size_t capacity;           // power-of-two
    size_t mask;
};

struct Region {
    int fd = -1;
    size_t length = 0;
    void* addr = nullptr;
};

static Region map_region(const std::string& name, size_t bytes, bool create) {
    int flags = O_RDWR;
    if (create) flags |= O_CREAT;
    int fd = shm_open(name.c_str(), flags, 0600);
    if (fd < 0) throw std::runtime_error("shm_open failed");
    if (create) {
        if (ftruncate(fd, bytes) != 0) {
            close(fd);
            shm_unlink(name.c_str());
            throw std::runtime_error("ftruncate failed");
        }
    }
    void* addr = mmap(nullptr, bytes, PROT_READ|PROT_WRITE, MAP_SHARED, fd, 0);
    if (addr == MAP_FAILED) {
        close(fd);
        if (create) shm_unlink(name.c_str());
        throw std::runtime_error("mmap failed");
    }
    // optional: mlock(addr, bytes); madvise(addr, bytes, MADV_HUGEPAGE);
    return {fd, bytes, addr};
}

class SPSC {
public:
    SPSC(const std::string& name, size_t capacity_bytes, bool create) : name_(name) {
        size_t cap = next_pow2(capacity_bytes);
        size_t total = sizeof(Header) + cap;
        reg_ = map_region(name, total, create);

        hdr_ = reinterpret_cast<Header*>(reg_.addr);
        data_ = reinterpret_cast<unsigned char*>(reinterpret_cast<char*>(reg_.addr) + sizeof(Header));

        if (create) {
            new (&hdr_->head) std::atomic<size_t>(0);
            new (&hdr_->tail) std::atomic<size_t>(0);
            hdr_->capacity = cap;
            hdr_->mask = cap - 1;
        } else {
            // sanity checks could be added
        }
    }

    ~SPSC() {
        if (reg_.addr) munmap(reg_.addr, reg_.length);
        if (reg_.fd >= 0) close(reg_.fd);
        // do not unlink here; let the owner decide
    }

    size_t capacity() const { return hdr_->capacity; }
    size_t size() const {
        size_t h = hdr_->head.load(std::memory_order_acquire);
        size_t t = hdr_->tail.load(std::memory_order_acquire);
        return t - h;
    }
    size_t free_space() const { return hdr_->capacity - size(); }

    // Message framing: [uint32_t len][bytes ...]
    // Returns true if written, false if not enough space.
    bool try_push(const char* src, size_t len) {
        if (len > (hdr_->capacity - 4)) return false;
        size_t need = 4 + len;

        size_t head = hdr_->head.load(std::memory_order_acquire);
        size_t tail = hdr_->tail.load(std::memory_order_relaxed);
        size_t used = tail - head;
        if ((hdr_->capacity - used) < need) return false; // not enough space

        // write length (little-endian)
        write_bytes(reinterpret_cast<const unsigned char*>(&len), 4, tail);
        tail += 4;
        // write payload
        write_bytes(reinterpret_cast<const unsigned char*>(src), len, tail);
        tail += len;

        hdr_->tail.store(tail, std::memory_order_release);
        return true;
    }

    // Returns py::bytes if available, else None
    py::object try_pop() {
        size_t head = hdr_->head.load(std::memory_order_relaxed);
        size_t tail = hdr_->tail.load(std::memory_order_acquire);
        if (tail - head < 4) return py::none(); // not enough for length

        // read length
        uint32_t len = 0;
        read_bytes(reinterpret_cast<unsigned char*>(&len), 4, head);
        if (tail - head < 4 + len) return py::none(); // incomplete frame

        head += 4;
        std::string out;
        out.resize(len);
        read_bytes(reinterpret_cast<unsigned char*>(&out[0]), len, head);
        head += len;

        hdr_->head.store(head, std::memory_order_release);
        return py::bytes(out);
    }

    // helpers
    void write_bytes(const unsigned char* src, size_t n, size_t pos) {
        size_t cap = hdr_->capacity;
        size_t i = pos & hdr_->mask;
        size_t first = std::min(n, cap - i);
        std::memcpy(&data_[i], src, first);
        if (n > first) std::memcpy(&data_[0], src + first, n - first);
    }
    void read_bytes(unsigned char* dst, size_t n, size_t pos) {
        size_t cap = hdr_->capacity;
        size_t i = pos & hdr_->mask;
        size_t first = std::min(n, cap - i);
        std::memcpy(dst, &data_[i], first);
        if (n > first) std::memcpy(dst + first, &data_[0], n - first);
    }

    static void unlink(const std::string& name) {
        shm_unlink(name.c_str()); // ignore errors
    }

private:
    std::string name_;
    Region reg_{};
    Header* hdr_ = nullptr;
    unsigned char* data_ = nullptr;
};

PYBIND11_MODULE(fastmq, m) {
    py::class_<SPSC>(m, "SPSC")
        .def_static("unlink", &SPSC::unlink)
        .def(py::init<const std::string&, size_t, bool>(), py::arg("name"), py::arg("capacity_bytes"), py::arg("create"))
        .def("capacity", &SPSC::capacity)
        .def("size", &SPSC::size)
        .def("try_push", [](SPSC& q, py::bytes b){
            char* buf; py::ssize_t len;
            if (PYBIND11_BYTES_AS_STRING_AND_SIZE(b.ptr(), &buf, &len)) throw std::runtime_error("bytes extract failed");
            return q.try_push(buf, (size_t)len);
        })
        .def("try_pop", &SPSC::try_pop);
}
