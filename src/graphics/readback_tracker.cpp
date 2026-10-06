/**
 * ReXGlue runtime - lazy readback of GPU-written guest memory.
 */

#include <rex/graphics/readback_tracker.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iterator>
#include <vector>

#include <rex/assert.h>
#include <rex/logging.h>

namespace rex::graphics {

ReadbackTracker::ReadbackTracker(memory::Memory& memory, AwaitSubmission await_submission,
                                 PrepareSource prepare_source)
    : memory_(memory),
      await_submission_(std::move(await_submission)),
      prepare_source_(std::move(prepare_source)),
      page_size_(uint32_t(rex::memory::page_size())) {
  provider_handle_ = memory_.RegisterPhysicalMemoryDataProvider(ProvideThunk, this);
}

ReadbackTracker::~ReadbackTracker() {
  // The readback buffers go away with the command processor.
  FlushAll();
  memory_.UnregisterPhysicalMemoryDataProvider(provider_handle_);
}

void ReadbackTracker::AddPendingRange(uint32_t physical_address, uint32_t length,
                                      const uint8_t* data, void* source_handle,
                                      uint64_t submission) {
  if (!length) {
    return;
  }
  uint32_t end = physical_address + length;
  auto global_lock = global_critical_region_.Acquire();

  // Supersede the overlapping parts of older pending ranges.
  auto it = pending_.upper_bound(physical_address);
  if (it != pending_.begin()) {
    auto previous = std::prev(it);
    if (previous->second.end > physical_address) {
      it = previous;
    }
  }
  while (it != pending_.end() && it->first < end) {
    uint32_t old_start = it->first;
    PendingRange old_range = it->second;
    it = pending_.erase(it);
    if (old_start < physical_address) {
      PendingRange left = old_range;
      left.end = physical_address;
      pending_.emplace(old_start, left);
    }
    if (old_range.end > end) {
      PendingRange right = old_range;
      right.data = old_range.data + (end - old_start);
      it = pending_.emplace(end, right).first;
      ++it;
    }
  }
  pending_.emplace(physical_address, PendingRange{end, data, source_handle, submission});
  ++stats_.ranges_added;

  // Any access to these pages must make the data current first. Invalidation
  // watches are already set by SharedMemory::RangeWrittenByGpu.
  memory_.EnablePhysicalMemoryAccessCallbacks(physical_address, length, false, true);
}

void ReadbackTracker::AwaitForRange(const PendingRange& range) {
  if (range.submission <= submission_awaited_) {
    return;
  }
  auto wait_start = std::chrono::steady_clock::now();
  if (!await_submission_(range.submission)) {
    REXGPU_ERROR("ReadbackTracker: failed to await submission {}", range.submission);
  }
  ++stats_.gpu_waits;
  stats_.wait_microseconds += uint64_t(std::chrono::duration_cast<std::chrono::microseconds>(
                                           std::chrono::steady_clock::now() - wait_start)
                                           .count());
  submission_awaited_ = range.submission;
}

void ReadbackTracker::WriteRange(uint32_t start, const PendingRange& range) {
  if (prepare_source_) {
    prepare_source_(range.source_handle);
  }
  // ROV research: REX_VK_AB_READBACK_SABOTAGE=1 writes the data 4 bytes off -
  // the negative control of the lazy readback A/B check.
  static const bool ab_sabotage = [] {
    const char* env = std::getenv("REX_VK_AB_READBACK_SABOTAGE");
    return env && *env == '1';
  }();
  uint32_t length = range.end - start;
  if (ab_sabotage && length > 4) {
    std::memcpy(memory_.TranslatePhysical(start) + 4, range.data, length - 4);
  } else {
    std::memcpy(memory_.TranslatePhysical(start), range.data, length);
  }
  ++stats_.ranges_provided;
  stats_.bytes_provided += range.end - start;
}

std::pair<uint32_t, uint32_t> ReadbackTracker::ProvideThunk(void* context_ptr,
                                                            uint32_t physical_address_start,
                                                            uint32_t length, bool discard) {
  return reinterpret_cast<ReadbackTracker*>(context_ptr)
      ->Provide(physical_address_start, length, discard);
}

std::pair<uint32_t, uint32_t> ReadbackTracker::Provide(uint32_t physical_address_start,
                                                       uint32_t length, bool discard) {
  // Called by the memory system with the global critical region locked.
  ++stats_.provider_calls;
  if (!length || pending_.empty()) {
    return std::make_pair(physical_address_start, length);
  }
  // All the pending ranges touching the pages of the requested range are made
  // current, and then all the ranges touching their pages, and so on, because
  // the data providers are disabled for whole pages.
  uint64_t page_mask = uint64_t(page_size_) - 1;
  uint64_t range_start = uint64_t(physical_address_start) & ~page_mask;
  uint64_t range_end = (uint64_t(physical_address_start) + length + page_mask) & ~page_mask;
  bool range_grown = true;
  while (range_grown) {
    range_grown = false;
    auto it = pending_.upper_bound(uint32_t(range_start));
    if (it != pending_.begin()) {
      auto previous = std::prev(it);
      if (previous->second.end > range_start) {
        it = previous;
      }
    }
    for (; it != pending_.end() && it->first < range_end; ++it) {
      uint64_t pending_start = uint64_t(it->first) & ~page_mask;
      uint64_t pending_end = (uint64_t(it->second.end) + page_mask) & ~page_mask;
      if (pending_start < range_start) {
        range_start = pending_start;
        range_grown = true;
      }
      if (pending_end > range_end) {
        range_end = pending_end;
        range_grown = true;
      }
    }
  }

  auto it = pending_.upper_bound(uint32_t(range_start));
  if (it != pending_.begin()) {
    auto previous = std::prev(it);
    if (previous->second.end > range_start) {
      it = previous;
    }
  }
  while (it != pending_.end() && it->first < range_end) {
    if (!discard) {
      AwaitForRange(it->second);
      WriteRange(it->first, it->second);
    }
    it = pending_.erase(it);
  }
  return std::make_pair(uint32_t(range_start),
                        uint32_t(std::min(range_end - range_start, uint64_t(UINT32_MAX))));
}

void ReadbackTracker::FlushSource(const uint8_t* source_begin, size_t source_size) {
  auto global_lock = global_critical_region_.Acquire();
  const uint8_t* source_end = source_begin + source_size;
  std::vector<std::pair<uint32_t, uint32_t>> flushed;
  for (auto it = pending_.begin(); it != pending_.end();) {
    if (it->second.data >= source_begin && it->second.data < source_end) {
      flushed.emplace_back(it->first, it->second.end - it->first);
      ++it;
    } else {
      ++it;
    }
  }
  for (const auto& range : flushed) {
    std::pair<uint32_t, uint32_t> provided = Provide(range.first, range.second, false);
    memory_.DisablePhysicalMemoryDataProviders(provided.first, provided.second);
  }
}

void ReadbackTracker::FlushAll() {
  auto global_lock = global_critical_region_.Acquire();
  while (!pending_.empty()) {
    auto it = pending_.begin();
    std::pair<uint32_t, uint32_t> provided =
        Provide(it->first, it->second.end - it->first, false);
    memory_.DisablePhysicalMemoryDataProviders(provided.first, provided.second);
  }
}

ReadbackTracker::Stats ReadbackTracker::TakeStats() {
  auto global_lock = global_critical_region_.Acquire();
  Stats stats = stats_;
  stats_ = Stats();
  return stats;
}

}  // namespace rex::graphics
