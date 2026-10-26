// GPU allocator for PyTorch backed by CUDA unified (managed) memory.
//
// Built by super_processor.managed_memory with nvcc and loaded through
// torch.cuda.memory.CUDAPluggableAllocator, only in the FlashVSR child that
// runs when a clip does not fit in free GPU memory. Managed blocks live on
// the GPU while there is room and the driver pages the rest to system RAM,
// so running short of VRAM costs speed instead of failing.
//
// Freed blocks are kept and handed out again, as PyTorch's own allocator
// does: asking the driver for a managed block on every allocation made
// FlashVSR three to four times slower even with the GPU half empty. A
// request takes the smallest cached block that fits, up to twice its size.
// Reusing only blocks of the exact size left half of what was held idle
// (13.6 of 27.6 GiB for one 72-frame preview), so whenever more than
// CACHE_SLACK sits idle at a miss, the whole cache goes back to the driver.
//
// Everything taken from the driver, in use or cached, is held under the
// byte limit in SUPER_PROCESSOR_MANAGED_LIMIT. Without one, the pages the
// driver moved to system RAM had no ceiling: one preview took 80 GiB and the
// kernel began killing programs. Past the limit the cache is returned to the
// driver first, and if the block still does not fit the allocation throws
// std::bad_alloc, which reaches Python as MemoryError. It must throw:
// PyTorch does not check what this returns, and a null block came back as
// a tensor at address 0 whose first write broke the CUDA context.
#include <cuda_runtime.h>
#include <sys/types.h>

#include <cstdlib>
#include <map>
#include <mutex>
#include <new>
#include <unordered_map>

namespace {

// Idle cache allowed before a miss returns it all to the driver.
const size_t CACHE_SLACK = size_t(2) << 30;

std::mutex cache_lock;
std::multimap<size_t, void*> cache;          // idle blocks by size
std::unordered_map<void*, size_t> sizes;     // every block held, by address
size_t held = 0;    // bytes taken from the driver, in use or cached
size_t cached = 0;  // of those, bytes idle in the cache

// 2 MiB granules, so sizes that differ by a little share blocks.
size_t rounded(ssize_t size) {
    const size_t granule = size_t(2) << 20;
    return (size_t(size) + granule - 1) / granule * granule;
}

// Read once. No limit set means no allocation at all: the child is never
// started without one, and a stray load must not page RAM without bound.
size_t limit() {
    static const size_t value = [] {
        const char* raw = std::getenv("SUPER_PROCESSOR_MANAGED_LIMIT");
        return raw ? size_t(std::strtoull(raw, nullptr, 10)) : size_t(0);
    }();
    return value;
}

// Give every idle block back to the driver. Caller holds cache_lock.
void release_cache() {
    for (auto& entry : cache) {
        cudaFree(entry.second);
        sizes.erase(entry.second);
        held -= entry.first;
    }
    cache.clear();
    cached = 0;
}

}  // namespace

extern "C" {

void* managed_malloc(ssize_t size, int device, cudaStream_t /*stream*/) {
    const size_t want = rounded(size);
    std::lock_guard<std::mutex> guard(cache_lock);
    auto found = cache.lower_bound(want);
    if (found != cache.end() && found->first <= 2 * want) {
        void* block = found->second;
        cached -= found->first;
        cache.erase(found);
        return block;
    }
    if (cached > CACHE_SLACK || held + want > limit()) {
        release_cache();
    }
    if (held + want > limit()) {
        throw std::bad_alloc();
    }
    void* block = nullptr;
    if (cudaMallocManaged(&block, want, cudaMemAttachGlobal) != cudaSuccess) {
        cudaGetLastError();  // clear it, so later calls do not see it
        throw std::bad_alloc();
    }
    held += want;
    sizes[block] = want;
    // Keep pages on this GPU while they fit, and map them for it even when
    // they have moved to system RAM, so a page is not bounced on every touch.
#if CUDART_VERSION >= 13000
    cudaMemLocation gpu{};
    gpu.type = cudaMemLocationTypeDevice;
    gpu.id = device;
#else
    const int gpu = device;
#endif
    cudaMemAdvise(block, want, cudaMemAdviseSetPreferredLocation, gpu);
    cudaMemAdvise(block, want, cudaMemAdviseSetAccessedBy, gpu);
    return block;
}

void managed_free(void* block, ssize_t /*size*/, int /*device*/, cudaStream_t /*stream*/) {
    std::lock_guard<std::mutex> guard(cache_lock);
    // The block may be larger than this request asked for; use its own size.
    const size_t own = sizes.at(block);
    cache.emplace(own, block);
    cached += own;
}

}  // extern "C"
