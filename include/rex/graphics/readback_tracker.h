/**
 * ReXGlue runtime - lazy readback of GPU-written guest memory.
 */

#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <utility>

#include <rex/memory.h>
#include <rex/thread/mutex.h>

namespace rex::graphics {

// Lazy readback (readback_resolve = "lazy"): guest memory ranges written by
// the GPU (resolves, memory export) whose data is copied from host-visible
// readback buffers to the guest memory only when the CPU accesses those pages
// (memory data providers, which make the pages inaccessible meanwhile). The
// guest sees exactly the data of the immediate ("full") readback, but the
// command processor never waits for the GPU at a resolve.
//
// Pending ranges never overlap: a newer range supersedes the overlapping parts
// of older ones. Everything is protected by the global critical region, the
// same lock the memory system holds while calling the data provider.
class ReadbackTracker {
 public:
  // Waits from any thread until a submission (already submitted) is completed
  // by the GPU.
  using AwaitSubmission = std::function<bool(uint64_t submission)>;
  // Makes the host-visible readback memory identified by the handle visible to
  // the CPU (invalidation of non-coherent memory), after the GPU wrote it.
  using PrepareSource = std::function<void(void* source_handle)>;

  ReadbackTracker(memory::Memory& memory, AwaitSubmission await_submission,
                  PrepareSource prepare_source);
  ~ReadbackTracker();
  ReadbackTracker(const ReadbackTracker&) = delete;
  ReadbackTracker& operator=(const ReadbackTracker&) = delete;

  // `length` bytes at `data` (host-visible memory identified by
  // `source_handle`) will hold the GPU data of the guest physical range once
  // `submission`, which must already be submitted, is completed.
  void AddPendingRange(uint32_t physical_address, uint32_t length, const uint8_t* data,
                       void* source_handle, uint64_t submission);

  // Writes all pending data whose source lies in [source_begin,
  // source_begin + source_size) to the guest memory - before the readback
  // buffer is destroyed or reallocated.
  void FlushSource(const uint8_t* source_begin, size_t source_size);
  // Writes all pending data to the guest memory.
  void FlushAll();

  struct Stats {
    uint64_t provider_calls = 0;
    uint64_t ranges_provided = 0;
    uint64_t bytes_provided = 0;
    uint64_t gpu_waits = 0;
    uint64_t wait_microseconds = 0;
    uint64_t ranges_added = 0;
  };
  // Returns the counters since the previous call and resets them.
  Stats TakeStats();
  size_t pending_range_count() const { return pending_.size(); }

 private:
  struct PendingRange {
    uint32_t end;
    const uint8_t* data;
    void* source_handle;
    uint64_t submission;
  };

  static std::pair<uint32_t, uint32_t> ProvideThunk(void* context_ptr,
                                                    uint32_t physical_address_start,
                                                    uint32_t length, bool discard);
  std::pair<uint32_t, uint32_t> Provide(uint32_t physical_address_start, uint32_t length,
                                        bool discard);
  // Writes the data of the pending range (the GPU must be done with it).
  void WriteRange(uint32_t start, const PendingRange& range);
  // Awaits the submission if it's newer than the latest awaited one.
  void AwaitForRange(const PendingRange& range);

  memory::Memory& memory_;
  AwaitSubmission await_submission_;
  PrepareSource prepare_source_;
  void* provider_handle_ = nullptr;
  uint32_t page_size_;

  rex::thread::global_critical_region global_critical_region_;
  // By the start address, non-overlapping.
  std::map<uint32_t, PendingRange> pending_;
  // The newest submission known to be completed (only grows).
  uint64_t submission_awaited_ = 0;

  Stats stats_;
};

}  // namespace rex::graphics
