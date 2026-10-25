// GPU allocator for PyTorch backed by CUDA unified (managed) memory.
//
// Built by super_processor.managed_memory with nvcc and loaded through
// torch.cuda.memory.CUDAPluggableAllocator, only in the FlashVSR child that
// runs when a clip does not fit in free GPU memory. Managed blocks live on
// the GPU while there is room and the driver pages the rest to system RAM,
// so running short of VRAM costs speed instead of failing.
//
// Freed blocks are kept and handed out again for the same rounded size, as
// PyTorch's own allocator does. Asking the driver for a managed block on
// every allocation made FlashVSR three to four times slower even with the
// GPU half empty; with the cache it is within 1.5x of the normal allocator.
// Nothing goes back to the driver until the child exits.
#include <cuda_runtime.h>
#include <sys/types.h>

#include <mutex>
#include <unordered_map>
#include <vector>

namespace {

std::mutex cache_lock;
std::unordered_map<size_t, std::vector<void*>> cache;

// 2 MiB granules, so sizes that differ by a little share blocks.
size_t rounded(ssize_t size) {
    const size_t granule = size_t(2) << 20;
    return (size_t(size) + granule - 1) / granule * granule;
}

}  // namespace

extern "C" {

void* managed_malloc(ssize_t size, int device, cudaStream_t /*stream*/) {
    const size_t want = rounded(size);
    {
        std::lock_guard<std::mutex> guard(cache_lock);
        auto found = cache.find(want);
        if (found != cache.end() && !found->second.empty()) {
            void* block = found->second.back();
            found->second.pop_back();
            return block;
        }
    }
    void* block = nullptr;
    if (cudaMallocManaged(&block, want, cudaMemAttachGlobal) != cudaSuccess) {
        return nullptr;
    }
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

void managed_free(void* block, ssize_t size, int /*device*/, cudaStream_t /*stream*/) {
    std::lock_guard<std::mutex> guard(cache_lock);
    cache[rounded(size)].push_back(block);
}

}  // extern "C"
