#define NOMINMAX
#include <cstdint>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <unordered_map>
#include <cstring>
#include <complex>
#include <utility>
#include <list>
#include <atomic>
#include <condition_variable>
#include <queue>
#include <thread>
#include <cstddef>
#include <unordered_set>
#include <memory>
#include <tuple>
#include <immintrin.h> // For prefetch and SIMD intrinsics (if available)
#include <deque> // For std::queue underlying container
#include <array>
#include <pybind11/complex.h>
#include <pybind11/stl.h>
#include <optional>
#include <future>
#include <cmath> // For std::cos, std::sin, std::acos
#include <vector>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#undef min
#undef max

namespace py = pybind11;

#if defined(__GNUC__) && !defined(__clang__)
#define NOOPTIMIZE [[gnu::noinline, gnu::nooptimize]]
#elif defined(_MSC_VER)
#define NOOPTIMIZE __declspec(noinline)
#pragma optimize("", off)
#else
#define NOOPTIMIZE
#endif

#if defined(_MSC_VER)
#define HOTPATH_KERNEL __declspec(noinline)
#else
#define HOTPATH_KERNEL [[gnu::noinline, gnu::nooptimize]]
#endif

// AVX2 kernel for AND+carry-save (matmul)
HOTPATH_KERNEL
inline void avx2_and_carrysave_kernel(const uint64_t* a, const uint64_t* b, __m256i& sum, __m256i& carry, size_t n) {
#if defined(__AVX2__) && defined(__x86_64__)
#if defined(__GNUC__) && !defined(__clang__)
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        __asm__ __volatile__ (
            "vmovdqu   (%[a]), %%ymm0\n\t"
            "vmovdqu   (%[b]), %%ymm1\n\t"
            "vpand     %%ymm1, %%ymm0, %%ymm2\n\t"
            "vpxor     %%ymm2, %[sum], %%ymm3\n\t"
            "vpxor     %%ymm3, %[carry], %%ymm3\n\t"
            "vpand     %[sum], %%ymm2, %%ymm4\n\t"
            "vpand     %[sum], %[carry], %%ymm5\n\t"
            "vpand     %%ymm2, %[carry], %%ymm6\n\t"
            "vpor      %%ymm4, %%ymm5, %%ymm4\n\t"
            "vpor      %%ymm4, %%ymm6, %%ymm4\n\t"
            "vmovdqu   %%ymm3, %[sum]\n\t"
            "vmovdqu   %%ymm4, %[carry]\n\t"
            : [sum] "+m" (sum), [carry] "+m" (carry)
            : [a] "r" (a + i), [b] "r" (b + i)
            : "ymm0", "ymm1", "ymm2", "ymm3", "ymm4", "ymm5", "ymm6", "memory"
        );
    }
    for (; i < n; ++i) {
        uint64_t x = a[i] & b[i];
        uint64_t s = ((uint64_t*)&sum)[i % 4] ^ x ^ ((uint64_t*)&carry)[i % 4];
        uint64_t c = (((uint64_t*)&sum)[i % 4] & x) | (((uint64_t*)&sum)[i % 4] & ((uint64_t*)&carry)[i % 4]) | (x & ((uint64_t*)&carry)[i % 4]);
        ((uint64_t*)&sum)[i % 4] = s;
        ((uint64_t*)&carry)[i % 4] = c;
    }
#elif defined(_MSC_VER)
    #pragma optimize("", off)
    for (size_t i = 0; i + 4 <= n; i += 4) {
        __m256i ymm_a = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + i));
        __m256i ymm_b = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + i));
        __m256i x = _mm256_and_si256(ymm_a, ymm_b);
        __m256i newsum = _mm256_xor_si256(_mm256_xor_si256(sum, x), carry);
        __m256i newcarry = _mm256_or_si256(_mm256_and_si256(sum, x), _mm256_or_si256(_mm256_and_si256(sum, carry), _mm256_and_si256(x, carry)));
        sum = newsum;
        carry = newcarry;
    }
    for (size_t i = (n / 4) * 4; i < n; ++i) {
        uint64_t x = a[i] & b[i];
        uint64_t s = ((uint64_t*)&sum)[i % 4] ^ x ^ ((uint64_t*)&carry)[i % 4];
        uint64_t c = (((uint64_t*)&sum)[i % 4] & x) | (((uint64_t*)&sum)[i % 4] & ((uint64_t*)&carry)[i % 4]) | (x & ((uint64_t*)&carry)[i % 4]);
        ((uint64_t*)&sum)[i % 4] = s;
        ((uint64_t*)&carry)[i % 4] = c;
    }
    #pragma optimize("", on)
#else
    for (size_t i = 0; i + 4 <= n; i += 4) {
        __m256i ymm_a = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + i));
        __m256i ymm_b = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + i));
        __m256i x = _mm256_and_si256(ymm_a, ymm_b);
        __m256i newsum = _mm256_xor_si256(_mm256_xor_si256(sum, x), carry);
        __m256i newcarry = _mm256_or_si256(_mm256_and_si256(sum, x), _mm256_or_si256(_mm256_and_si256(sum, carry), _mm256_and_si256(x, carry)));
        sum = newsum;
        carry = newcarry;
    }
    for (size_t i = (n / 4) * 4; i < n; ++i) {
        uint64_t x = a[i] & b[i];
        uint64_t s = ((uint64_t*)&sum)[i % 4] ^ x ^ ((uint64_t*)&carry)[i % 4];
        uint64_t c = (((uint64_t*)&sum)[i % 4] & x) | (((uint64_t*)&sum)[i % 4] & ((uint64_t*)&carry)[i % 4]) | (x & ((uint64_t*)&carry)[i % 4]);
        ((uint64_t*)&sum)[i % 4] = s;
        ((uint64_t*)&carry)[i % 4] = c;
    }
#endif
#else
    for (size_t i = 0; i < n; ++i) {
        uint64_t x = a[i] & b[i];
        uint64_t s = ((uint64_t*)&sum)[i % 4] ^ x ^ ((uint64_t*)&carry)[i % 4];
        uint64_t c = (((uint64_t*)&sum)[i % 4] & x) | (((uint64_t*)&sum)[i % 4] & ((uint64_t*)&carry)[i % 4]) | (x & ((uint64_t*)&carry)[i % 4]);
        ((uint64_t*)&sum)[i % 4] = s;
        ((uint64_t*)&carry)[i % 4] = c;
    }
#endif
}

// AVX2 kernel for fused bitplane ops (AND/OR/XOR)
HOTPATH_KERNEL
inline void avx2_fused_bitplane_ops_kernel(const uint64_t* a, const uint64_t* b, uint64_t* out, size_t n, int op) {
#if defined(__AVX2__)
#if defined(_MSC_VER)
    #pragma optimize("", off)
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        __m256i va = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + i));
        __m256i vb = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + i));
        __m256i vout;
        if (op == 0) vout = _mm256_and_si256(va, vb);
        else if (op == 1) vout = _mm256_or_si256(va, vb);
        else vout = _mm256_xor_si256(va, vb);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(out + i), vout);
    }
    for (; i < n; ++i) {
        if (op == 0) out[i] = a[i] & b[i];
        else if (op == 1) out[i] = a[i] | b[i];
        else out[i] = a[i] ^ b[i];
    }
    #pragma optimize("", on)
#elif defined(__GNUC__) && !defined(__clang__)
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        __asm__ __volatile__ (
            "vmovdqu   (%[a]), %%ymm0\n\t"
            "vmovdqu   (%[b]), %%ymm1\n\t"
            "cmpl      $0, %[op]\n\t"
            "jne       1f\n\t"
            "vpand     %%ymm1, %%ymm0, %%ymm2\n\t"
            "jmp       3f\n\t"
            "1: cmpl   $1, %[op]\n\t"
            "jne       2f\n\t"
            "vpor      %%ymm1, %%ymm0, %%ymm2\n\t"
            "jmp       3f\n\t"
            "2: vpxor  %%ymm1, %%ymm0, %%ymm2\n\t"
            "3: vmovdqu %%ymm2, (%[out])\n\t"
            :
            : [a] "r" (a + i), [b] "r" (b + i), [out] "r" (out + i), [op] "r" (op)
            : "ymm0", "ymm1", "ymm2", "memory"
        );
    }
    for (; i < n; ++i) {
        if (op == 0) out[i] = a[i] & b[i];
        else if (op == 1) out[i] = a[i] | b[i];
        else out[i] = a[i] ^ b[i];
    }
#else
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        __m256i va = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + i));
        __m256i vb = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + i));
        __m256i vout;
        if (op == 0) vout = _mm256_and_si256(va, vb);
        else if (op == 1) vout = _mm256_or_si256(va, vb);
        else vout = _mm256_xor_si256(va, vb);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(out + i), vout);
    }
    for (; i < n; ++i) {
        if (op == 0) out[i] = a[i] & b[i];
        else if (op == 1) out[i] = a[i] | b[i];
        else out[i] = a[i] ^ b[i];
    }
#endif
#else
    for (size_t i = 0; i < n; ++i) {
        if (op == 0) out[i] = a[i] & b[i];
        else if (op == 1) out[i] = a[i] | b[i];
        else out[i] = a[i] ^ b[i];
    }
#endif
}

// AVX2 kernel for bitplane packing (decomposition)
HOTPATH_KERNEL
inline void avx2_pack_bitplanes(const double* XPtr, uint64_t* OutPtr, size_t N) {
#if defined(__AVX2__)
    size_t idx = 0;
    for (; idx + 4 <= N; idx += 4) {
        __m256d v = _mm256_loadu_pd(XPtr + idx);
        alignas(32) double vals[4];
        _mm256_store_pd(vals, v);
        for (size_t j = 0; j < 4; ++j) {
            uint64_t Val;
            memcpy(&Val, &vals[j], sizeof(Val));
            OutPtr[idx + j] = Val;
        }
    }
    for (; idx < N; ++idx) {
        uint64_t Val;
        memcpy(&Val, &XPtr[idx], sizeof(Val));
        OutPtr[idx] = Val;
    }
#else
    for (size_t idx = 0; idx < N; ++idx) {
        uint64_t Val;
        memcpy(&Val, &XPtr[idx], sizeof(Val));
        OutPtr[idx] = Val;
    }
#endif
}

// AVX2 kernel for bitplane unpacking (recomposition)
HOTPATH_KERNEL
inline void avx2_unpack_bitplanes(const uint64_t* BPPtr, double* OutPtr, size_t N) {
#if defined(__AVX2__)
    size_t idx = 0;
    for (; idx + 4 <= N; idx += 4) {
        alignas(32) double dvals[4];
        for (size_t j = 0; j < 4; ++j) memcpy(&dvals[j], &BPPtr[idx + j], sizeof(double));
        __m256d v = _mm256_load_pd(dvals);
        _mm256_storeu_pd(OutPtr + idx, v);
    }
    for (; idx < N; ++idx) {
        double dval;
        memcpy(&dval, &BPPtr[idx], sizeof(double));
        OutPtr[idx] = dval;
    }
#else
    for (size_t idx = 0; idx < N; ++idx) {
        double dval;
        memcpy(&dval, &BPPtr[idx], sizeof(double));
        OutPtr[idx] = dval;
    }
#endif
}

int NumThreads() {
    return std::thread::hardware_concurrency();
}

#if defined(_WIN32) || defined(_WIN64)
    #include <BaseTsd.h>
    typedef SSIZE_T ssize_t;
    #include <windows.h>
    #include <processthreadsapi.h>
#elif defined(__linux__)
    #include <numa.h>
    #include <sched.h>
    #include <unistd.h>
#endif

static constexpr int MAX_SLIDING_WINDOW_SIZE = 64 * 1024; // 64 kB

namespace NUMAUtils {
    inline int get_numa_node_count() {
    #if defined(_WIN32) || defined(_WIN64)
        ULONG highestNode = 0;
        if (GetNumaHighestNodeNumber(&highestNode)) return int(highestNode + 1);
        return 1;
    #elif defined(__linux__)
        if (numa_available() < 0) return 1;
        return numa_max_node() + 1;
    #else
        return 1;
    #endif
    }
    inline int get_numa_node_for_thread(int thread_idx, int num_threads) {
        int nodes = get_numa_node_count();
        return nodes > 1 ? (thread_idx % nodes) : 0;
    }
    inline void bind_thread_to_numa_node(int thread_idx, int numa_node) {
    #if defined(_WIN32) || defined(_WIN64)
        GROUP_AFFINITY affinity = {0};
        affinity.Mask = 1ULL << (thread_idx % 64);
        affinity.Group = 0;
        SetThreadGroupAffinity(GetCurrentThread(), &affinity, nullptr);
    #elif defined(__linux__)
        if (numa_available() < 0) return;
        bitmask* bm = numa_allocate_nodemask();
        numa_bitmask_clearall(bm);
        numa_bitmask_setbit(bm, numa_node);
        numa_bind(bm);
        numa_free_nodemask(bm);
    #else
        (void)thread_idx; (void)numa_node;
    #endif
    }
}

// LZ4Compress: Compress a Python bytes or str object using a simple LZ4-like algorithm
py::bytes LZ4Compress(py::bytes Data) {
    std::string DataStr = (std::string)Data;
    if (DataStr.empty()) return py::bytes("");
    constexpr int MinMatch = 4;
    std::string Output;
    std::unordered_map<uint32_t, int> Hashes;
    int I = 0;
    while (I < static_cast<int>(DataStr.size())) {
        int BestLen = 0, BestOffset = 0;
        if (I + MinMatch <= static_cast<int>(DataStr.size())) {
            uint32_t Hash = 0;
            for (int k = 0; k < MinMatch; ++k)
                Hash = (Hash * 257) + static_cast<uint8_t>(DataStr[I + k]);
            auto it = Hashes.find(Hash);
            if (it != Hashes.end() && I - it->second <= MAX_SLIDING_WINDOW_SIZE) {
                int J = it->second;
                int Len = 0;
                while (I + Len < static_cast<int>(DataStr.size()) && DataStr[J + Len] == DataStr[I + Len] && Len < 255) ++Len;
                if (Len >= MinMatch) {
                    BestLen = Len;
                    BestOffset = I - J;
                }
            }
            Hashes[Hash] = I;
        }
        if (BestLen >= MinMatch) {
            Output.push_back(0); // Marker for match
            Output.push_back(static_cast<char>(BestLen));
            Output.push_back(static_cast<char>(BestOffset >> 8));
            Output.push_back(static_cast<char>(BestOffset & 0xFF));
            I += BestLen;
        } else {
            Output.push_back(DataStr[I]);
            ++I;
        }
    }
    return py::bytes(Output);
}

// LZ4Decompress: Decompress a Python bytes or str object using a simple LZ4-like algorithm
py::bytes LZ4Decompress(py::bytes Data) {
    std::string DataStr = (std::string)Data;
    if (DataStr.empty()) return py::bytes("");
    std::string Output;
    for (size_t I = 0; I < DataStr.size();) {
        if (static_cast<unsigned char>(DataStr[I]) == 0 && I + 3 < DataStr.size()) {
            int Len = static_cast<uint8_t>(DataStr[I + 1]);
            int Offset = (static_cast<uint8_t>(DataStr[I + 2]) << 8) | static_cast<uint8_t>(DataStr[I + 3]);
            if (Offset > 0 && Output.size() >= static_cast<size_t>(Offset)) {
                size_t Start = Output.size() - Offset;
                for (int k = 0; k < Len; ++k)
                    Output.push_back(Output[Start + k]);
                I += 4;
                continue;
            }
        }
        Output.push_back(DataStr[I]);
        ++I;
    }
    return py::bytes(Output);
}

// Fast bit-mixing hash for tensor/array content with SIMD-friendly path for large arrays and prefetching
std::size_t HashTensor(py::array arr) {
    py::buffer_info info = arr.request();
    const uint8_t* data = static_cast<const uint8_t*>(info.ptr);
    size_t nbytes = info.size * info.itemsize;
    std::size_t hash = 0x9e3779b97f4a7c15ull; // Large prime seed
    constexpr size_t kSIMDWidth = 32; // 256 bits
    if (nbytes >= 256 && reinterpret_cast<uintptr_t>(data) % kSIMDWidth == 0) {
        // SIMD-friendly, process 32 bytes at a time
        const __m256i* data256 = reinterpret_cast<const __m256i*>(data);
        size_t n256 = nbytes / kSIMDWidth;
        for (size_t i = 0; i < n256; ++i) {
            // Prefetch next cache line
            if (i + 2 < n256) _mm_prefetch(reinterpret_cast<const char*>(data256 + i + 2), _MM_HINT_T0);
            __m256i chunk = _mm256_load_si256(data256 + i);
            alignas(32) uint64_t vals[4];
            _mm256_store_si256((__m256i*)vals, chunk);
            for (int j = 0; j < 4; ++j) {
                hash ^= (vals[j] + 0x9e3779b9 + (hash << 6) + (hash >> 2));
                hash = (hash | (hash << 13)) ^ (hash >> 7);
            }
        }
        for (size_t i = n256 * kSIMDWidth; i < nbytes; ++i) {
            hash ^= (static_cast<std::size_t>(data[i]) + 0x9e3779b9 + (hash << 6) + (hash >> 2));
            hash = (hash | (hash << 13)) ^ (hash >> 7);
        }
    } else {
        for (size_t i = 0; i < nbytes; ++i) {
            if (i + 64 < nbytes) _mm_prefetch(reinterpret_cast<const char*>(data + i + 64), _MM_HINT_T0);
            hash ^= (static_cast<std::size_t>(data[i]) + 0x9e3779b9 + (hash << 6) + (hash >> 2));
            hash = (hash | (hash << 13)) ^ (hash >> 7);
        }
    }
    hash ^= (hash << 21);
    hash ^= (hash >> 17);
    hash ^= (hash << 31);
    return hash;
}

constexpr size_t L1_CACHE_SIZE = 128;
struct alignas(64) L1CacheEntry {
    std::atomic<size_t> hash{0};
    py::object value;
    std::atomic<bool> valid{false};
};
class LockFreeL1Cache {
public:
    LockFreeL1Cache() {
        for (auto& entry : entries) entry.valid.store(false, std::memory_order_relaxed);
    }
    bool get(size_t hash, py::object& out) {
        for (size_t i = 0; i < L1_CACHE_SIZE; ++i) {
            _mm_prefetch(reinterpret_cast<const char*>(&entries[i]), _MM_HINT_T0);
            if (entries[i].valid.load(std::memory_order_acquire) && entries[i].hash.load(std::memory_order_relaxed) == hash) {
                out = entries[i].value;
                return true;
            }
        }
        return false;
    }
    void put(size_t hash, const py::object& value) {
        for (size_t i = 0; i < L1_CACHE_SIZE; ++i) {
            if (!entries[i].valid.load(std::memory_order_acquire)) {
                entries[i].hash.store(hash, std::memory_order_relaxed);
                entries[i].value = value;
                entries[i].valid.store(true, std::memory_order_release);
                return;
            }
        }
        // If full, evict random
        size_t idx = hash % L1_CACHE_SIZE;
        entries[idx].hash.store(hash, std::memory_order_relaxed);
        entries[idx].value = value;
        entries[idx].valid.store(true, std::memory_order_release);
    }
    void clear() {
        for (auto& entry : entries) entry.valid.store(false, std::memory_order_relaxed);
    }
private:
    std::array<L1CacheEntry, L1_CACHE_SIZE> entries;
};

// Lock-free ring buffer for async writes
class LockFreeRingBuffer {
public:
    explicit LockFreeRingBuffer(size_t capacity) : buffer_(capacity), head_(0), tail_(0), size_(0) {}
    bool push(const std::pair<std::size_t, py::object>& item) {
        size_t cur_tail = tail_.load(std::memory_order_relaxed);
        size_t next_tail = (cur_tail + 1) % buffer_.size();
        if (next_tail == head_.load(std::memory_order_acquire)) return false; // full
        buffer_[cur_tail] = item;
        tail_.store(next_tail, std::memory_order_release);
        size_.fetch_add(1, std::memory_order_relaxed);
        return true;
    }
    std::optional<std::pair<std::size_t, py::object>> pop() {
        size_t cur_head = head_.load(std::memory_order_relaxed);
        if (cur_head == tail_.load(std::memory_order_acquire)) return std::nullopt; // empty
        auto item = buffer_[cur_head];
        head_.store((cur_head + 1) % buffer_.size(), std::memory_order_release);
        size_.fetch_sub(1, std::memory_order_relaxed);
        return item;
    }
    size_t size() const { return size_.load(std::memory_order_relaxed); }
private:
    std::vector<std::pair<std::size_t, py::object>> buffer_;
    std::atomic<size_t> head_;
    std::atomic<size_t> tail_;
    std::atomic<size_t> size_;
};

class TensorCache {
public:
    TensorCache(size_t MaxSize) : MaxSize_(MaxSize), PendingWrites_(1024) {
        LRUList_.reserve(MaxSize_);
        Bitmap_.reserve(MaxSize_);
        WorkerRunning_.store(false, std::memory_order_relaxed);
    }
private:
    size_t MaxSize_;
    std::vector<std::size_t> LRUList_; // Use vector for better cache locality
    py::dict Cache_; // hash -> value
    py::dict AccessCount_; // hash -> int
    std::vector<uint8_t> Bitmap_;
    py::dict BitmapLookup_; // hash -> idx
    LockFreeRingBuffer PendingWrites_;
    std::atomic<bool> WorkerRunning_;
    std::thread WorkerThread_;
    LockFreeL1Cache L1Cache_;

    void Worker() {
        while (WorkerRunning_.load(std::memory_order_acquire)) {
            for (int i = 0; i < 32; ++i) {
                auto item = PendingWrites_.pop();
                if (!item) break;
                auto [hash, value] = *item;
                PutImpl(hash, value);
            }
            std::this_thread::sleep_for(std::chrono::microseconds(100));
        }
    }

    void PutImpl(std::size_t hash, const py::object& value) {
        L1Cache_.put(hash, value);
        py::int_ pykey(hash);
        if (Cache_.contains(pykey)) {
            // Move to front (index 0)
            auto it = std::find(LRUList_.begin(), LRUList_.end(), hash);
            if (it != LRUList_.end() && it != LRUList_.begin()) {
                std::rotate(LRUList_.begin(), it, it + 1);
            }
            LRUList_[0] = hash;
            int count = 1;
            if (AccessCount_.contains(pykey)) {
                count = int(AccessCount_.attr("get")(pykey, py::int_(0)).cast<int>()) + 1;
            }
            AccessCount_.attr("__setitem__")(pykey, py::int_(count));
            SetBitmap(hash, 1);
            return;
        }
        // Insert at front
        LRUList_.insert(LRUList_.begin(), hash);
        Cache_.attr("__setitem__")(pykey, value);
        AccessCount_.attr("__setitem__")(pykey, py::int_(1));
        SetBitmap(hash, 1);
        if (py::len(Cache_) > MaxSize_) {
            std::size_t OldHash = LRUList_.back();
            LRUList_.pop_back();
            py::int_ oldkey(OldHash);
            Cache_.attr("pop")(oldkey);
            AccessCount_.attr("pop")(oldkey);
            RemoveBitmap(OldHash);
        }
    }

    void SetBitmap(std::size_t hash, uint8_t Value) {
        py::int_ pykey(hash);
        if (!BitmapLookup_.contains(pykey)) {
            size_t idx = Bitmap_.size();
            BitmapLookup_.attr("__setitem__")(pykey, py::int_(idx));
            Bitmap_.push_back(Value);
        } else {
            size_t idx = BitmapLookup_.attr("get")(pykey, py::int_(0)).cast<size_t>();
            Bitmap_[idx] = Value;
        }
    }

    void RemoveBitmap(std::size_t hash) {
        py::int_ pykey(hash);
        if (BitmapLookup_.contains(pykey)) {
            size_t idx = BitmapLookup_.attr("get")(pykey, py::int_(0)).cast<size_t>();
            if (idx < Bitmap_.size()) Bitmap_[idx] = 0;
        }
    }
public:
    void Put(const py::object& tensor) {
        std::size_t hash = HashTensor(tensor);
        while (!PendingWrites_.push({hash, tensor})) {
            std::this_thread::sleep_for(std::chrono::microseconds(10));
        }
    }

    py::object Get(const py::object& tensor) {
        std::size_t hash = HashTensor(tensor);
        py::object l1val;
        if (L1Cache_.get(hash, l1val)) return l1val;
        py::int_ pykey(hash);
        if (!Cache_.contains(pykey)) return py::object();
        // Move to front (index 0)
        auto it = std::find(LRUList_.begin(), LRUList_.end(), hash);
        if (it != LRUList_.end() && it != LRUList_.begin()) {
            std::rotate(LRUList_.begin(), it, it + 1);
        }
        LRUList_[0] = hash;
        int count = 1;
        if (AccessCount_.contains(pykey)) {
            count = int(AccessCount_.attr("get")(pykey, py::int_(0)).cast<int>()) + 1;
        }
        AccessCount_.attr("__setitem__")(pykey, py::int_(count));
        SetBitmap(hash, 1);
        L1Cache_.put(hash, Cache_.attr("get")(pykey, py::object()));
        return Cache_.attr("get")(pykey, py::object());
    }

    void Clear() {
        L1Cache_.clear();
        Cache_.attr("clear")();
        LRUList_.clear();
        AccessCount_.attr("clear")();
        Bitmap_.clear();
        BitmapLookup_.attr("clear")();
        // No need to clear PendingWrites_ (lock-free ring buffer)
    }

    size_t Size() {
        return py::len(Cache_);
    }

    void StartWorker() {
        WorkerRunning_.store(true, std::memory_order_release);
        WorkerThread_ = std::thread(&TensorCache::Worker, this);
    }

    void StopWorker() {
        WorkerRunning_.store(false, std::memory_order_release);
        if (WorkerThread_.joinable()) WorkerThread_.join();
    }
};

inline int CountBits(size_t X) {
    int Count = 0;
    while (X) { Count += X & 1; X >>= 1; }
    return Count;
}

// DecomposeBitplanes: input float64 array, output uint64_t array (same shape)
py::array_t<uint64_t> DecomposeBitplanes(py::array_t<double> X) {
    py::buffer_info Info = X.request();
    std::vector<size_t> OutShape(Info.shape.begin(), Info.shape.end());
    std::vector<ssize_t> OutShapePy;
    OutShapePy.reserve(OutShape.size());
    for (size_t v : OutShape) OutShapePy.push_back(static_cast<ssize_t>(v));
    auto Out = py::array_t<uint64_t>(OutShapePy);
    uint64_t* OutPtr = static_cast<uint64_t*>(Out.request().ptr);
    double* XPtr = static_cast<double*>(Info.ptr);
    size_t N = 1;
    for (size_t i = 0; i < Info.shape.size(); ++i) N *= Info.shape[i];
    avx2_pack_bitplanes(XPtr, OutPtr, N);
    return Out;
}

// RecomposeBitplanes: input uint64_t array, output float64 array (original shape)
py::array_t<double> RecomposeBitplanes(py::array_t<uint64_t> BP) {
    py::buffer_info Info = BP.request();
    std::vector<size_t> OutShape(Info.shape.begin(), Info.shape.end());
    std::vector<ssize_t> OutShapePy;
    OutShapePy.reserve(OutShape.size());
    for (size_t v : OutShape) OutShapePy.push_back(static_cast<ssize_t>(v));
    auto Out = py::array_t<double>(OutShapePy);
    double* OutPtr = static_cast<double*>(Out.request().ptr);
    uint64_t* BPPtr = static_cast<uint64_t*>(Info.ptr);
    size_t N = 1;
    for (size_t i = 0; i < OutShape.size(); ++i) N *= OutShape[i];
    avx2_unpack_bitplanes(BPPtr, OutPtr, N);
    return Out;
}

// SIMD-friendly carry-save adder for two uint64_t vectors
inline void carry_save_add(__m256i& sum, __m256i& carry, const __m256i& x) {
    __m256i new_sum = _mm256_xor_si256(_mm256_xor_si256(sum, x), carry);
    __m256i new_carry = _mm256_or_si256(_mm256_and_si256(sum, x), _mm256_or_si256(_mm256_and_si256(sum, carry), _mm256_and_si256(x, carry)));
    sum = new_sum;
    carry = new_carry;
}

// Scalar carry-save adder for two uint64_t
inline void carry_save_add(uint64_t& sum, uint64_t& carry, uint64_t x) {
    uint64_t new_sum = sum ^ x ^ carry;
    uint64_t new_carry = (sum & x) | (sum & carry) | (x & carry);
    sum = new_sum;
    carry = new_carry;
}

// Use kernels in hotpaths
py::array_t<uint64_t> FastBitplaneMatmul(py::array_t<uint64_t> A, py::array_t<uint64_t> B) {
    py::buffer_info InfoA = A.request();
    py::buffer_info InfoB = B.request();
    if (InfoA.ndim != 2 || InfoB.ndim != 2) throw std::runtime_error("Inputs must be 2D (M,N) and (N,P)");
    size_t M = InfoA.shape[0], N = InfoA.shape[1], P = InfoB.shape[1];
    py::array_t<uint64_t> Out(std::vector<ssize_t>{static_cast<ssize_t>(M), static_cast<ssize_t>(P)});
    uint64_t* OutPtr = static_cast<uint64_t*>(Out.request().ptr);
    uint64_t* APtr = static_cast<uint64_t*>(InfoA.ptr);
    uint64_t* BPtr = static_cast<uint64_t*>(InfoB.ptr);
#if defined(__AVX2__)
    constexpr size_t kSIMD = 4;
    for (size_t m = 0; m < M; ++m) {
        for (size_t p = 0; p < P; ++p) {
            __m256i sum = _mm256_setzero_si256();
            __m256i carry = _mm256_setzero_si256();
            avx2_and_carrysave_kernel(APtr + (m * N), BPtr + (p), sum, carry, N);
            uint64_t sum_arr[kSIMD], carry_arr[kSIMD];
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(sum_arr), sum);
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(carry_arr), carry);
            uint64_t result = 0;
            for (size_t i = 0; i < kSIMD; ++i) result += sum_arr[i] + (carry_arr[i] << 1);
            OutPtr[m * P + p] = result;
        }
    }
#else
    for (size_t m = 0; m < M; ++m) {
        for (size_t p = 0; p < P; ++p) {
            uint64_t sum = 0, carry = 0;
            for (size_t n = 0; n < N; ++n) {
                uint64_t x = APtr[(m * N + n)] & BPtr[(n * P + p)];
                carry_save_add(sum, carry, x);
            }
            OutPtr[m * P + p] = sum + (carry << 1);
        }
    }
#endif
    return Out;
}

py::array_t<uint64_t> FusedBitplaneOps(py::array_t<uint64_t> A, py::array_t<uint64_t> B, std::string heuristic = "auto") {
    py::buffer_info InfoA = A.request();
    py::buffer_info InfoB = B.request();
    if (InfoA.size != InfoB.size) throw std::runtime_error("Input shapes must match");
    py::array_t<uint64_t> Out(InfoA.shape);
    uint64_t* OutPtr = static_cast<uint64_t*>(Out.request().ptr);
    uint64_t* APtr = static_cast<uint64_t*>(InfoA.ptr);
    uint64_t* BPtr = static_cast<uint64_t*>(InfoB.ptr);
    size_t N = InfoA.size;
    int op = 0;
    if (heuristic == "and") op = 0;
    else if (heuristic == "or") op = 1;
    else if (heuristic == "xor") op = 2;
    else op = 0;
    avx2_fused_bitplane_ops_kernel(APtr, BPtr, OutPtr, N, op);
    return Out;
}

py::array_t<uint64_t> FastBitplaneIntegerMatmul(py::array_t<uint64_t> A, py::array_t<uint64_t> B) {
    py::buffer_info InfoA = A.request();
    py::buffer_info InfoB = B.request();
    if (InfoA.ndim != 2 || InfoB.ndim != 2) throw std::runtime_error("Inputs must be 2D (M,N) and (N,P)");
    size_t M = InfoA.shape[0], N = InfoA.shape[1], P = InfoB.shape[1];
    py::array_t<uint64_t> Out(std::vector<ssize_t>{static_cast<ssize_t>(M), static_cast<ssize_t>(P)});
    uint64_t* OutPtr = static_cast<uint64_t*>(Out.request().ptr);
    uint64_t* APtr = static_cast<uint64_t*>(InfoA.ptr);
    uint64_t* BPtr = static_cast<uint64_t*>(InfoB.ptr);
    for (size_t m = 0; m < M; ++m) {
        for (size_t p = 0; p < P; ++p) {
            uint64_t sum = 0, carry = 0;
            for (size_t n = 0; n < N; ++n) {
                uint64_t x = APtr[(m * N + n)] * BPtr[(n * P + p)];
                carry_save_add(sum, carry, x);
            }
            OutPtr[m * P + p] = sum + (carry << 1);
        }
    }
    return Out;
}

// CliffordDotProduct: Clifford algebra dot product using bitwise sign computation
py::object CliffordDotProduct(py::array_t<double> A, py::array_t<double> B) {
    py::buffer_info InfoA = A.request();
    py::buffer_info InfoB = B.request();
    if (InfoA.ndim != 1 || InfoB.ndim != 1 || InfoA.size != InfoB.size)
        throw std::runtime_error("Inputs must be 1D arrays of the same size");
    size_t N = InfoA.size;
    const double* APtr = static_cast<const double*>(InfoA.ptr);
    const double* BPtr = static_cast<const double*>(InfoB.ptr);
    std::complex<double> Result(0.0, 0.0);
    double real_acc = 0.0;

    #if defined(__AVX2__) && defined(__POPCNT__)
    __m256d acc_vec = _mm256_setzero_pd();
    size_t i = 0;
    for (; i + 4 <= N; i += 4) {
        alignas(32) double signs[4];
        for (int j = 0; j < 4; ++j) {
            signs[j] = (_mm_popcnt_u64(i + j) & 1) ? -1.0 : 1.0;
        }
        __m256d sign_vec = _mm256_load_pd(signs);
        __m256d a_vals = _mm256_loadu_pd(&APtr[i]);
        __m256d b_vals = _mm256_loadu_pd(&BPtr[i]);
        acc_vec = _mm256_fmadd_pd(_mm256_mul_pd(a_vals, b_vals), sign_vec, acc_vec);
    }
    alignas(32) double acc_arr[4];
    _mm256_store_pd(acc_arr, acc_vec);
    real_acc += acc_arr[0] + acc_arr[1] + acc_arr[2] + acc_arr[3];
    for (; i < N; ++i) {
        int Sign = (_mm_popcnt_u64(i) & 1) ? -1 : 1;
        real_acc += Sign * APtr[i] * BPtr[i];
    }
    #else
    for (size_t i = 0; i < N; ++i) {
        int Sign = (CountBits(i) & 1) ? -1 : 1;
        real_acc += Sign * APtr[i] * BPtr[i];
    }
    #endif
    Result.real(real_acc);
    return py::cast(Result);
}

// Parallelized CliffordMatrixMultiply
py::array_t<std::complex<double>> CliffordMatrixMultiply(py::array_t<double> A, py::array_t<double> B) {
    py::buffer_info InfoA = A.request();
    py::buffer_info InfoB = B.request();
    if (InfoA.ndim != 2 || InfoB.ndim != 2 || InfoA.shape[1] != InfoB.shape[0])
        throw std::runtime_error("Inputs must be 2D arrays with compatible shapes");
    size_t M = InfoA.shape[0], K = InfoA.shape[1], N = InfoB.shape[1];
    const double* APtr = static_cast<const double*>(InfoA.ptr);
    const double* BPtr = static_cast<const double*>(InfoB.ptr);
    py::array_t<std::complex<double>> Out(std::vector<ssize_t>{static_cast<ssize_t>(M), static_cast<ssize_t>(N)});
    auto OutPtr = static_cast<std::complex<double>*>(Out.request().ptr);

    int num_threads = NumThreads();
    std::vector<std::future<void>> futures;
    size_t chunk_size = (M + num_threads - 1) / num_threads;

    for (int t = 0; t < num_threads; ++t) {
        size_t start_m = t * chunk_size;
        size_t end_m = std::min(start_m + chunk_size, M);
        if (start_m >= end_m) continue;

        futures.push_back(std::async(std::launch::async, [=]() {
            py::gil_scoped_release release;
            NUMAUtils::bind_thread_to_numa_node(t, NUMAUtils::get_numa_node_for_thread(t, num_threads));
            constexpr size_t TILE_M = 8, TILE_N = 8, TILE_K = 16;
            for (size_t mm = start_m; mm < end_m; mm += TILE_M) {
                for (size_t nn = 0; nn < N; nn += TILE_N) {
                    alignas(64) std::complex<double> acc_tile[TILE_M][TILE_N] = {{0.0}};
                    for (size_t kk = 0; kk < K; kk += TILE_K) {
                        for (size_t m_tile = 0; m_tile < TILE_M; ++m_tile) {
                            size_t m = mm + m_tile;
                            if (m >= end_m) continue;
                            for (size_t k_tile = 0; k_tile < TILE_K; ++k_tile) {
                                size_t k = kk + k_tile;
                                if (k >= K) continue;
                                _mm_prefetch(reinterpret_cast<const char*>(&BPtr[k * N + nn]), _MM_HINT_T0);
                                #if defined(__AVX2__) && defined(__POPCNT__)
                                size_t n_tile = 0;
                                __m256d a_val = _mm256_set1_pd(APtr[m * K + k]);
                                for (; n_tile + 4 <= TILE_N; n_tile += 4) {
                                    size_t n = nn + n_tile;
                                    if (n + 3 >= N) break;
                                    alignas(32) int64_t indices[4] = { (int64_t)(m + k + n), (int64_t)(m + k + n + 1), (int64_t)(m + k + n + 2), (int64_t)(m + k + n + 3) };
                                    alignas(32) double signs[4];
                                    for(int i=0; i<4; ++i) signs[i] = (_mm_popcnt_u64(indices[i]) & 1) ? -1.0 : 1.0;
                                    __m256d b_vals = _mm256_loadu_pd(&BPtr[k * N + n]);
                                    __m256d sign_vec = _mm256_load_pd(signs);
                                    double acc_reals[4], acc_imags[4];
                                    for(int i=0; i<4; ++i) { acc_reals[i] = acc_tile[m_tile][n_tile+i].real(); acc_imags[i] = acc_tile[m_tile][n_tile+i].imag(); }
                                    __m256d acc_re = _mm256_loadu_pd(acc_reals);
                                    acc_re = _mm256_fmadd_pd(_mm256_mul_pd(a_val, b_vals), sign_vec, acc_re);
                                    _mm256_storeu_pd(acc_reals, acc_re);
                                    for(int i=0; i<4; ++i) acc_tile[m_tile][n_tile+i] = std::complex<double>(acc_reals[i], acc_imags[i]);
                                }
                                for (; n_tile < TILE_N; ++n_tile) {
                                    size_t n = nn + n_tile;
                                    if (n < N) {
                                        int sign = ((CountBits(m + k + n) & 1) ? -1 : 1);
                                        acc_tile[m_tile][n_tile] += sign * APtr[m * K + k] * BPtr[k * N + n];
                                    }
                                }
                                #else
                                for (size_t n_tile = 0; n_tile < TILE_N; ++n_tile) {
                                    size_t n = nn + n_tile;
                                    if (n < N) {
                                        int sign = ((CountBits(m + k + n) & 1) ? -1 : 1);
                                        acc_tile[m_tile][n_tile] += sign * APtr[m * K + k] * BPtr[k * N + n];
                                    }
                                }
                                #endif
                            }
                        }
                    }
                    for (size_t m_tile = 0; m_tile < TILE_M; ++m_tile) {
                        size_t m = mm + m_tile;
                        if (m >= end_m) continue;
                        for (size_t n_tile = 0; n_tile < TILE_N; ++n_tile) {
                            size_t n = nn + n_tile;
                            if (n < N) OutPtr[m * N + n] = acc_tile[m_tile][n_tile];
                        }
                    }
                }
            }
        }));
    }
    for (auto& f : futures) f.get();
    return Out;
}

// CliffordMatrixVectorMultiply: Clifford algebra matrix-vector multiplication
py::array_t<std::complex<double>> CliffordMatrixVectorMultiply(py::array_t<double> A, py::array_t<double> B) {
    py::buffer_info InfoA = A.request();
    py::buffer_info InfoB = B.request();
    if (InfoA.ndim != 2 || InfoB.ndim != 1 || InfoA.shape[1] != InfoB.size)
        throw std::runtime_error("Inputs must be 2D and 1D arrays with compatible shapes");
    size_t M = InfoA.shape[0], N = InfoA.shape[1];
    const double* APtr = static_cast<const double*>(InfoA.ptr);
    const double* BPtr = static_cast<const double*>(InfoB.ptr);
    py::array_t<std::complex<double>> Out(std::vector<ssize_t>{static_cast<ssize_t>(M)});
    auto OutPtr = static_cast<std::complex<double>*>(Out.request().ptr);

    int num_threads = NumThreads();
    std::vector<std::future<void>> futures;
    size_t chunk_size = (M + num_threads - 1) / num_threads;

    for (int t = 0; t < num_threads; ++t) {
        size_t start_m = t * chunk_size;
        size_t end_m = std::min(start_m + chunk_size, M);
        if (start_m >= end_m) continue;

        futures.push_back(std::async(std::launch::async, [=]() {
            py::gil_scoped_release release;
            NUMAUtils::bind_thread_to_numa_node(t, NUMAUtils::get_numa_node_for_thread(t, num_threads));
            for (size_t m = start_m; m < end_m; ++m) {
                std::complex<double> acc(0.0, 0.0);
                #if defined(__AVX2__) && defined(__POPCNT__)
                __m256d acc_vec = _mm256_setzero_pd();
                size_t n = 0;
                for (; n + 4 <= N; n += 4) {
                    _mm_prefetch(reinterpret_cast<const char*>(&APtr[m * N + n + 16]), _MM_HINT_T0);
                    _mm_prefetch(reinterpret_cast<const char*>(&BPtr[n + 16]), _MM_HINT_T0);
                    alignas(32) int64_t indices[4] = {(int64_t)(m + n), (int64_t)(m + n + 1), (int64_t)(m + n + 2), (int64_t)(m + n + 3)};
                    alignas(32) double signs[4];
                    for(int i=0; i<4; ++i) signs[i] = (_mm_popcnt_u64(indices[i]) & 1) ? -1.0 : 1.0;
                    __m256d a_vals = _mm256_loadu_pd(&APtr[m * N + n]);
                    __m256d b_vals = _mm256_loadu_pd(&BPtr[n]);
                    __m256d sign_vec = _mm256_load_pd(signs);
                    acc_vec = _mm256_fmadd_pd(_mm256_mul_pd(a_vals, b_vals), sign_vec, acc_vec);
                }
                alignas(32) double acc_arr[4];
                _mm256_store_pd(acc_arr, acc_vec);
                acc.real(acc_arr[0] + acc_arr[1] + acc_arr[2] + acc_arr[3]);
                for (; n < N; ++n) {
                    int sign = ((CountBits(m + n) & 1) ? -1 : 1);
                    acc += sign * APtr[m * N + n] * BPtr[n];
                }
                #else
                for (size_t n = 0; n < N; ++n) {
                    int sign = ((CountBits(m + n) & 1) ? -1 : 1);
                    acc += sign * APtr[m * N + n] * BPtr[n];
                }
                #endif
                OutPtr[m] = acc;
            }
        }));
    }
    for (auto& f : futures) f.get();
    return Out;
}

// GeometricProduct: geometric product for 2D float64 arrays, fully optimized with
// parallelism, NUMA-awareness, cache-tiling, prefetching, and alignment.
py::array_t<std::complex<double>> GeometricProduct(py::array_t<double> A, py::array_t<double> B) {
    py::buffer_info InfoA = A.request();
    py::buffer_info InfoB = B.request();

    if (InfoA.ndim != 2 || InfoB.ndim != 2 || InfoA.shape[1] != InfoB.shape[0])
        throw std::runtime_error("Inputs must be 2D arrays with compatible shapes");

    size_t M = InfoA.shape[0], K = InfoA.shape[1], N = InfoB.shape[1];
    const double* APtr = static_cast<const double*>(InfoA.ptr);
    const double* BPtr = static_cast<const double*>(InfoB.ptr);

    py::array_t<std::complex<double>> Out(std::vector<ssize_t>{static_cast<ssize_t>(M), static_cast<ssize_t>(N)});
    auto OutPtr = static_cast<std::complex<double>*>(Out.request().ptr);

    int num_threads = NumThreads();
    std::vector<std::future<void>> futures;

    size_t chunk_size = (M + num_threads - 1) / num_threads;

    for (int t = 0; t < num_threads; ++t) {
        size_t start_m = t * chunk_size;
        size_t end_m = std::min(start_m + chunk_size, M);

        if (start_m >= end_m) continue;

        futures.push_back(std::async(std::launch::async, [=]() {
            py::gil_scoped_release release;
            NUMAUtils::bind_thread_to_numa_node(t, NUMAUtils::get_numa_node_for_thread(t, num_threads));
            constexpr size_t TILE_M = 8;
            constexpr size_t TILE_N = 8;
            constexpr size_t TILE_K = 16;

            for (size_t mm = start_m; mm < end_m; mm += TILE_M) {
                for (size_t nn = 0; nn < N; nn += TILE_N) {
                    alignas(64) std::complex<double> acc_tile[TILE_M][TILE_N] = {{0.0}};
                    for (size_t kk = 0; kk < K; kk += TILE_K) {
                        if (kk + TILE_K < K) {
                            _mm_prefetch(reinterpret_cast<const char*>(&APtr[mm * K + (kk + TILE_K)]), _MM_HINT_T0);
                            _mm_prefetch(reinterpret_cast<const char*>(&BPtr[(kk + TILE_K) * N + nn]), _MM_HINT_T0);
                        }
                        for (size_t m_tile = 0; m_tile < TILE_M; ++m_tile) {
                            for (size_t k_tile = 0; k_tile < TILE_K; ++k_tile) {
                                size_t m = mm + m_tile;
                                size_t k = kk + k_tile;
                                if (m < end_m && k < K) {
#if defined(__AVX2__)
                                    size_t n_tile = 0;
                                    for (; n_tile + 4 <= TILE_N; n_tile += 4) {
                                        size_t n0 = nn + n_tile;
                                        if (n0 + 3 < N) {
                                            _mm_prefetch(reinterpret_cast<const char*>(&acc_tile[m_tile][n_tile]), _MM_HINT_T0);
                                            __m256d a_val = _mm256_set1_pd(APtr[m * K + k]);
                                            __m256d b_val = _mm256_loadu_pd(&BPtr[k * N + n0]);
                                            __m256d acc_re = _mm256_setzero_pd();
                                            __m256d acc_im = _mm256_setzero_pd();
                                            // Load current acc_tile values
                                            double reals[4], imags[4];
                                            for (int i = 0; i < 4; ++i) {
                                                reals[i] = acc_tile[m_tile][n_tile + i].real();
                                                imags[i] = acc_tile[m_tile][n_tile + i].imag();
                                            }
                                            acc_re = _mm256_loadu_pd(reals);
                                            acc_im = _mm256_loadu_pd(imags);
                                            // Complex multiply (real only, as input is double)
                                            acc_re = _mm256_add_pd(acc_re, _mm256_mul_pd(a_val, b_val));
                                            // Store back
                                            _mm256_storeu_pd(reals, acc_re);
                                            for (int i = 0; i < 4; ++i) acc_tile[m_tile][n_tile + i].real(reals[i]);
                                            // Imaginary part remains zero
                                        }
                                    }
                                    for (; n_tile < TILE_N; ++n_tile) {
                                        size_t n = nn + n_tile;
                                        if (n < N) {
                                            acc_tile[m_tile][n_tile] += APtr[m * K + k] * BPtr[k * N + n];
                                        }
                                    }
#endif
                                }
                            }
                        }
                    }
                    for (size_t m_tile = 0; m_tile < TILE_M; ++m_tile) {
                        for (size_t n_tile = 0; n_tile < TILE_N; ++n_tile) {
                            size_t m = mm + m_tile;
                            size_t n = nn + n_tile;
                            if (m < end_m && n < N) {
                                _mm_prefetch(reinterpret_cast<const char*>(&OutPtr[m * N + n]), _MM_HINT_T0);
                                OutPtr[m * N + n] = acc_tile[m_tile][n_tile];
                            }
                        }
                    }
                }
            }
        }));
    }
    for (auto& f : futures) {
        f.get();
    }
    return Out;
}

// Helper: compute sign for blades
inline int BladeSign(size_t a1, size_t b1) {
    int n1 = CountBits(a1 & b1);
    return (n1 & 1) ? -1 : 1;
}

// FastInnerProductBitwise: bitwise inner product for geometric algebra
py::array_t<std::complex<double>> FastInnerProductBitwise(py::array_t<std::complex<double>> A, py::array_t<std::complex<double>> B, int Dim) {
    py::buffer_info InfoA = A.request();
    py::buffer_info InfoB = B.request();
    size_t NBlades = 1ULL << Dim;
    if (InfoA.size != (ssize_t)NBlades || InfoB.size != (ssize_t)NBlades)
        throw std::runtime_error("Input array sizes must match 2^Dim");

    auto Out = py::array_t<std::complex<double>>(std::vector<ssize_t>{static_cast<ssize_t>(NBlades)});
    auto OutPtr = static_cast<std::complex<double>*>(Out.request().ptr);
    std::fill(OutPtr, OutPtr + NBlades, std::complex<double>(0.0, 0.0));
    auto APtr = static_cast<std::complex<double>*>(InfoA.ptr);
    auto BPtr = static_cast<std::complex<double>*>(InfoB.ptr);

    int num_threads = NumThreads();
    std::vector<std::vector<std::complex<double>>> thread_local_results(num_threads, std::vector<std::complex<double>>(NBlades, {0.0, 0.0}));
    std::vector<std::future<void>> futures;

    size_t chunk_size = (NBlades + num_threads - 1) / num_threads;

    for (int t = 0; t < num_threads; ++t) {
        size_t start_I = t * chunk_size;
        size_t end_I = std::min(start_I + chunk_size, NBlades);
        if (start_I >= end_I) continue;

        futures.push_back(std::async(std::launch::async, [=, &thread_local_results, &APtr, &BPtr]() {
            py::gil_scoped_release release;
            auto& local_out = thread_local_results[t];
            for (size_t I = start_I; I < end_I; ++I) {
                int grade_i = CountBits(I);
                for (size_t J = 0; J < NBlades; ++J) {
                    if (static_cast<size_t>(std::abs(grade_i - CountBits(J))) == CountBits(I ^ J)) {
                        size_t ResultBlade = I ^ J;
                        int Sign = BladeSign(I, J);
                        local_out[ResultBlade] += std::complex<double>(Sign) * APtr[I] * BPtr[J];
                    }
                }
            }
        }));
    }

    for (auto& f : futures) f.get();

    for (size_t i = 0; i < NBlades; ++i) {
        for (int t = 0; t < num_threads; ++t) {
            OutPtr[i] += thread_local_results[t][i];
        }
    }
    return Out;
}

// FastOuterProductBitwise: bitwise outer product for geometric algebra
py::array_t<std::complex<double>> FastOuterProductBitwise(py::array_t<std::complex<double>> A, py::array_t<std::complex<double>> B, int Dim) {
    py::buffer_info InfoA = A.request();
    py::buffer_info InfoB = B.request();
    size_t NBlades = 1ULL << Dim;
    if (InfoA.size != (ssize_t)NBlades || InfoB.size != (ssize_t)NBlades)
        throw std::runtime_error("Input array sizes must match 2^Dim");
    auto Out = py::array_t<std::complex<double>>(std::vector<ssize_t>{static_cast<ssize_t>(NBlades)});
    auto OutPtr = static_cast<std::complex<double>*>(Out.request().ptr);
    std::fill(OutPtr, OutPtr + NBlades, std::complex<double>(0.0, 0.0));
    auto APtr = static_cast<std::complex<double>*>(InfoA.ptr);
    auto BPtr = static_cast<std::complex<double>*>(InfoB.ptr);

    int num_threads = NumThreads();
    std::vector<std::vector<std::complex<double>>> thread_local_results(num_threads, std::vector<std::complex<double>>(NBlades, {0.0, 0.0}));
    std::vector<std::future<void>> futures;

    size_t chunk_size = (NBlades + num_threads - 1) / num_threads;

    for (int t = 0; t < num_threads; ++t) {
        size_t start_I = t * chunk_size;
        size_t end_I = std::min(start_I + chunk_size, NBlades);
        if (start_I >= end_I) continue;

        futures.push_back(std::async(std::launch::async, [=, &thread_local_results, &APtr, &BPtr]() {
            py::gil_scoped_release release;
            auto& local_out = thread_local_results[t];
            for (size_t I = start_I; I < end_I; ++I) {
                int grade_i = CountBits(I);
                for (size_t J = 0; J < NBlades; ++J) {
                    if ((I & J) == 0) { // Condition for outer product
                        if (grade_i + CountBits(J) <= static_cast<size_t>(Dim)) {
                            size_t ResultBlade = I | J;
                            int Sign = BladeSign(I, J);
                            local_out[ResultBlade] += std::complex<double>(Sign) * APtr[I] * BPtr[J];
                        }
                    }
                }
            }
        }));
    }

    for (auto& f : futures) f.get();

    for (size_t i = 0; i < NBlades; ++i) {
        for (int t = 0; t < num_threads; ++t) {
            OutPtr[i] += thread_local_results[t][i];
        }
    }
    return Out;
}

py::object DynamicTensorProcess(py::array arr) {
    size_t n = arr.size();
    if (n < 1024) {
        // Direct, no caching/compression
        return arr;
    } else if (n < 100000) {
        // Lazy cache and prefetch (simulate by returning arr for now)
        return arr;
    } else {
        // Full pipeline: decompose, quantize, compress
        auto decomp = DecomposeBitplanes(arr.cast<py::array_t<double>>());
        py::bytes comp = LZ4Compress(py::bytes(reinterpret_cast<const char*>(decomp.request().ptr), decomp.nbytes()));
        return comp;
    }
}

// RMTBitplaneMatmul: RMT-inspired, parallel, bitplane Clifford/geometric product
py::array_t<uint64_t> RMTBitplaneMatmul(py::array_t<uint64_t> A, py::array_t<uint64_t> B, std::string heuristic = "auto") {
    py::buffer_info InfoA = A.request();
    py::buffer_info InfoB = B.request();
    if (InfoA.ndim != 2 || InfoB.ndim != 2)
        throw std::runtime_error("Inputs must be 2D (M,N) and (N,P)");
    size_t M = InfoA.shape[0], N = InfoA.shape[1], P = InfoB.shape[1];
    py::array_t<uint64_t> Out(std::vector<ssize_t>{static_cast<ssize_t>(M), static_cast<ssize_t>(P)});
    uint64_t* OutPtr = static_cast<uint64_t*>(Out.request().ptr);
    uint64_t* APtr = static_cast<uint64_t*>(InfoA.ptr);
    uint64_t* BPtr = static_cast<uint64_t*>(InfoB.ptr);
    int num_threads = std::thread::hardware_concurrency();
    size_t chunk = (M + num_threads - 1) / num_threads;
    std::vector<std::future<void>> futures;
    for (int t = 0; t < num_threads; ++t) {
        size_t start = t * chunk;
        size_t end = std::min(M, (t + 1) * chunk);
        futures.push_back(std::async(std::launch::async, [=]() {
            py::gil_scoped_release release;
            for (size_t m = start; m < end; ++m) {
                for (size_t p = 0; p < P; ++p) {
                    uint64_t acc = 0;
                    for (size_t n = 0; n < N; ++n) {
                        int sign = ((CountBits(m + n + p) & 1) ? -1 : 1);
                        acc |= (APtr[m * N + n] & BPtr[n * P + p]);
                    }
                    OutPtr[m * P + p] = acc;
                }
            }
        }));
    }
    for (auto& f : futures) f.get();
    return Out;
}

// DecomposeComplexBitplanes: input complex<double> array, output uint64_t array (same shape + last dim = 2)
py::array_t<uint64_t> DecomposeComplexBitplanes(py::array_t<std::complex<double>> X) {
    py::buffer_info Info = X.request();
    std::vector<ssize_t> OutShape(Info.shape.begin(), Info.shape.end());
    OutShape.push_back(2); // real, imag
    auto Out = py::array_t<uint64_t>(OutShape);
    uint64_t* OutPtr = static_cast<uint64_t*>(Out.request().ptr);
    auto* XPtr = static_cast<std::complex<double>*>(Info.ptr);
    size_t N = X.size();
    int num_threads = NumThreads();
    size_t chunk_size = (N + num_threads - 1) / num_threads;
    std::vector<std::future<void>> futures;
    for (int t = 0; t < num_threads; ++t) {
        size_t start = t * chunk_size;
        size_t end = std::min(N, start + chunk_size);
        if (start >= end) continue;
        futures.push_back(std::async(std::launch::async, [=]() {
            py::gil_scoped_release release;
            for (size_t idx = start; idx < end; ++idx) {
                uint64_t real_bits, imag_bits;
                double real_part = XPtr[idx].real();
                double imag_part = XPtr[idx].imag();
                memcpy(&real_bits, &real_part, sizeof(real_bits));
                memcpy(&imag_bits, &imag_part, sizeof(imag_bits));
                OutPtr[idx * 2 + 0] = real_bits;
                OutPtr[idx * 2 + 1] = imag_bits;
            }
        }));
    }
    for(auto& f : futures) f.get();
    return Out;
}

// RecomposeComplexBitplanes: input uint64_t array (last dim=2), output complex<double> array
py::array_t<std::complex<double>> RecomposeComplexBitplanes(py::array_t<uint64_t> BP) {
    py::buffer_info Info = BP.request();
    if (Info.ndim == 0 || Info.shape.back() != 2) {
        throw std::runtime_error("Input must be at least 1D and the last dimension must be 2");
    }
    std::vector<ssize_t> OutShape(Info.shape.begin(), Info.shape.end() - 1);
    auto Out = py::array_t<std::complex<double>>(OutShape);
    auto* OutPtr = static_cast<std::complex<double>*>(Out.request().ptr);
    uint64_t* BPPtr = static_cast<uint64_t*>(Info.ptr);
    size_t N = Out.size();
    int num_threads = NumThreads();
    size_t chunk_size = (N + num_threads - 1) / num_threads;
    std::vector<std::future<void>> futures;
    for (int t = 0; t < num_threads; ++t) {
        size_t start = t * chunk_size;
        size_t end = std::min(N, start + chunk_size);
        if (start >= end) continue;
        futures.push_back(std::async(std::launch::async, [=]() {
            py::gil_scoped_release release;
            for (size_t idx = start; idx < end; ++idx) {
                double real_part, imag_part;
                memcpy(&real_part, &BPPtr[idx * 2 + 0], sizeof(real_part));
                memcpy(&imag_part, &BPPtr[idx * 2 + 1], sizeof(imag_part));
                OutPtr[idx] = std::complex<double>(real_part, imag_part);
            }
        }));
    }
    for(auto& f : futures) f.get();
    return Out;
}

// In-place iterative Cooley-Tukey FFT
void fft_inplace(std::vector<std::complex<double>>& a, bool invert) {
    int n = a.size();
    if (n <= 1) return;

    for (int i = 1, j = 0; i < n; i++) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) std::swap(a[i], a[j]);
    }

    for (int len = 2; len <= n; len <<= 1) {
        double ang = 2 * M_PI / len * (invert ? -1 : 1);
        std::complex<double> wlen(cos(ang), sin(ang));
        for (int i = 0; i < n; i += len) {
            std::complex<double> w(1);
            for (int j = 0; j < len / 2; j++) {
                std::complex<double> u = a[i + j];
                std::complex<double> v = a[i + j + len / 2] * w;
                a[i + j] = u + v;
                a[i + j + len / 2] = u - v;
                w *= wlen;
            }
        }
    }

    if (invert) {
        for (auto& x : a) x /= n;
    }
}

py::array_t<uint64_t> BitplaneFFT(py::array_t<uint64_t> BP, bool inverse) {
    auto recomposed_arr = RecomposeComplexBitplanes(BP);
    py::buffer_info info = recomposed_arr.request();
    auto* ptr = static_cast<std::complex<double>*>(info.ptr);
    if (info.ndim == 0) {
        return DecomposeComplexBitplanes(recomposed_arr);
    }
    size_t n_fft = info.shape.back();
    size_t n_vectors = recomposed_arr.size() / n_fft;
    int num_threads = NumThreads();
    size_t chunk_size = (n_vectors + num_threads - 1) / num_threads;
    std::vector<std::future<void>> futures;
    for (int t = 0; t < num_threads; ++t) {
        size_t start = t * chunk_size;
        size_t end = std::min(n_vectors, start + chunk_size);
        if (start >= end) continue;
        futures.push_back(std::async(std::launch::async, [=]() {
            py::gil_scoped_release release;
            for (size_t i = start; i < end; ++i) {
                std::vector<std::complex<double>> vec(ptr + i * n_fft, ptr + (i + 1) * n_fft);
                fft_inplace(vec, inverse);
                std::copy(vec.begin(), vec.end(), ptr + i * n_fft);
            }
        }));
    }
    for(auto& f : futures) f.get();
    return DecomposeComplexBitplanes(recomposed_arr);
}

py::array_t<uint64_t> BitplaneConvolution(py::array_t<uint64_t> BP_A, py::array_t<uint64_t> BP_K) {
    auto A = RecomposeComplexBitplanes(BP_A);
    auto K = RecomposeComplexBitplanes(BP_K);
    py::buffer_info infoA = A.request();
    py::buffer_info infoK = K.request();
    if (infoA.ndim != infoK.ndim) throw std::runtime_error("Input and kernel must have the same number of dimensions.");
    for(py::ssize_t i=0; i<infoA.ndim; ++i) {
        if (infoA.shape[i] != infoK.shape[i]) throw std::runtime_error("Input and kernel shapes must match.");
    }
    size_t n_fft = infoA.shape.back();
    auto A_fft = BitplaneFFT(BP_A, false);
    auto K_fft = BitplaneFFT(BP_K, false);
    auto recomposed_A_fft = RecomposeComplexBitplanes(A_fft);
    auto recomposed_K_fft = RecomposeComplexBitplanes(K_fft);
    auto* ptrA = static_cast<std::complex<double>*>(recomposed_A_fft.request().ptr);
    auto* ptrK = static_cast<std::complex<double>*>(recomposed_K_fft.request().ptr);
    size_t total_size = recomposed_A_fft.size();
    py::array_t<std::complex<double>> result_arr(infoA.shape);
    auto* ptr_res = static_cast<std::complex<double>*>(result_arr.request().ptr);
    for(size_t i = 0; i < total_size; ++i) {
        ptr_res[i] = ptrA[i] * ptrK[i];
    }
    auto result_bp = DecomposeComplexBitplanes(result_arr);
    return BitplaneFFT(result_bp, true);
}

// Helper: Compute the product of a vector of sizes
size_t prod(const std::vector<size_t>& dims) {
    size_t p = 1;
    for (size_t d : dims) p *= d;
    return p;
}

// Decompose ND arrays into a vector of 2D slices (views)
std::vector<std::pair<py::array, py::array>>
DecomposeNDTo2DBatches(py::array a, py::array b) {
    py::buffer_info info_a = a.request();
    py::buffer_info info_b = b.request();

    // Assume last two dims are matrix, rest are batch
    size_t a_ndim = info_a.ndim, b_ndim = info_b.ndim;
    if (a_ndim < 2 || b_ndim < 2)
        throw std::runtime_error("Inputs must be at least 2D");

    // Compute batch shape (broadcast)
    std::vector<size_t> batch_shape;
    size_t batch_dims = std::max(a_ndim, b_ndim) - 2;
    for (size_t i = 0; i < batch_dims; ++i) {
        size_t a_dim = (i < a_ndim - 2) ? info_a.shape[i] : 1;
        size_t b_dim = (i < b_ndim - 2) ? info_b.shape[i] : 1;
        if (a_dim != b_dim && a_dim != 1 && b_dim != 1)
            throw std::runtime_error("Batch dims not broadcastable");
        batch_shape.push_back(std::max(a_dim, b_dim));
    }
    size_t batch_size = prod(batch_shape);

    // Matrix dims
    size_t M = info_a.shape[a_ndim - 2];
    size_t K = info_a.shape[a_ndim - 1];
    size_t K2 = info_b.shape[b_ndim - 2];
    size_t N = info_b.shape[b_ndim - 1];
    if (K != K2)
        throw std::runtime_error("Matrix shapes do not align");

    // Prepare output: vector of pairs of 2D arrays
    std::vector<std::pair<py::array, py::array>> batches;
    batches.reserve(batch_size);

    // For each batch, create a view into the 2D slice
    for (size_t bidx = 0; bidx < batch_size; ++bidx) {
        size_t a_offset = bidx * M * K;
        size_t b_offset = bidx * K * N;
        // Use dtype from the original arrays
        py::array a2d = py::array(a.dtype(), {static_cast<ssize_t>(M), static_cast<ssize_t>(K)}, (char*)info_a.ptr + a_offset * info_a.itemsize);
        py::array b2d = py::array(b.dtype(), {static_cast<ssize_t>(K), static_cast<ssize_t>(N)}, (char*)info_b.ptr + b_offset * info_b.itemsize);
        batches.emplace_back(a2d, b2d);
    }
    return batches;
}

// Parallel batched Bitplane matmul
py::array BatchedBitplaneMatmul(py::array a, py::array b) {
    auto batches = DecomposeNDTo2DBatches(a, b);
    size_t batch_size = batches.size();
    std::vector<std::future<py::array_t<uint64_t>>> futures;
    for (size_t i = 0; i < batch_size; ++i) {
        futures.push_back(std::async(std::launch::async, [&, i]() {
            return FastBitplaneMatmul(batches[i].first.cast<py::array_t<uint64_t>>(), batches[i].second.cast<py::array_t<uint64_t>>());
        }));
    }
    std::vector<py::array_t<uint64_t>> results;
    for (auto& f : futures) {
        results.push_back(f.get());
    }
    // Stack results into ND array
    if (results.empty()) return py::array();
    auto shape = results[0].shape();
    std::vector<ssize_t> out_shape;
    out_shape.push_back(static_cast<ssize_t>(batch_size));
    for (size_t i = 0; i < results[0].ndim(); ++i)
        out_shape.push_back(static_cast<ssize_t>(shape[i]));
    py::array_t<uint64_t> out(out_shape);
    auto out_ptr = static_cast<uint64_t*>(out.request().ptr);
    size_t mat_size = results[0].size();
    for (size_t i = 0; i < batch_size; ++i) {
        auto in_ptr = static_cast<const uint64_t*>(results[i].request().ptr);
        std::memcpy(out_ptr + i * mat_size, in_ptr, mat_size * sizeof(uint64_t));
    }
    return out;
}

PYBIND11_MODULE(BitplaneEngine, m) {
    m.def("DecomposeBitplanes", &DecomposeBitplanes, "Decompose tensor into bitplanes");
    m.def("RecomposeBitplanes", &RecomposeBitplanes, "Recompose bitplanes into tensor");
    m.def("FastBitplaneMatmul", &FastBitplaneMatmul, "Bitplane matrix multiplication");
    m.def("FastBitplaneIntegerMatmul", &FastBitplaneIntegerMatmul, "Integer bitplane matrix multiplication");
    m.def("FastBitplaneOps", &FusedBitplaneOps, "Bitplane AND/OR/XOR ops");
    m.def("CliffordDotProduct", &CliffordDotProduct, "Clifford algebra dot product");
    m.def("CliffordMatrixMultiply", &CliffordMatrixMultiply, "Clifford algebra matrix multiplication");
    m.def("CliffordMatrixVectorMultiply", &CliffordMatrixVectorMultiply, "Clifford algebra matrix-vector multiplication");
    m.def("GeometricProduct", &GeometricProduct, "Geometric product for 2D arrays");
    m.def("FastInnerProductBitwise", &FastInnerProductBitwise, "Bitwise inner product for geometric algebra");
    m.def("FastOuterProductBitwise", &FastOuterProductBitwise, "Bitwise outer product for geometric algebra");
    m.def("HashTensor", &HashTensor, "Fast hash of a numpy array/tensor using bitwise tricks");
    m.def("LZ4Compress", &LZ4Compress, "Compress bytes using LZ4-like algorithm");
    m.def("LZ4Decompress", &LZ4Decompress, "Decompress bytes using LZ4-like algorithm");
    m.def("DynamicTensorProcess", &DynamicTensorProcess, "Dynamically process tensor based on size");
    m.def("RMTBitplaneMatmul", &RMTBitplaneMatmul, "RMT-inspired bitplane Clifford/geometric matmul");
    m.def("DecomposeComplexBitplanes", &DecomposeComplexBitplanes, "Decompose complex tensor into bitplanes");
    m.def("RecomposeComplexBitplanes", &RecomposeComplexBitplanes, "Recompose bitplanes into complex tensor");
    m.def("BitplaneFFT", &BitplaneFFT, "Perform FFT on bitplane-represented complex data", py::arg("BP"), py::arg("inverse") = false);
    m.def("BitplaneConvolution", &BitplaneConvolution, "Perform convolution on bitplane-represented complex data");
    m.def("DecomposeNDTo2DBatches", &DecomposeNDTo2DBatches, "Decompose ND arrays into 2D batches");
    m.def("BatchedBitplaneMatmul", &BatchedBitplaneMatmul, "Parallel batched bitplane matmul");

    py::class_<TensorCache>(m, "TensorCache")
        .def(py::init<size_t>())
        .def("Put", &TensorCache::Put)
        .def("Get", &TensorCache::Get)
        .def("Clear", &TensorCache::Clear)
        .def("Size", &TensorCache::Size)
        .def("StartWorker", &TensorCache::StartWorker)
        .def("StopWorker", &TensorCache::StopWorker);
}

