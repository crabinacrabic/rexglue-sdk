# Exact EDRAM emulation at full speed on AMD (Vulkan fragment shader interlock)

Research notes from the Viva Piñata recompilation ([VivaPinataRecomp](https://github.com/crabinacrabic/VivaPinataRecomp)),
October 2026. Branch `rov-research` of this fork; nothing here is upstream yet.

**Short version.** The exact EDRAM path (D3D12 rasterizer-ordered views / Vulkan fragment shader interlock, "ROV/FSI")
used to run Viva Piñata at 13–15 game ticks per second on an AMD Radeon RX 6600 (the game runs at 30). With the changes
below it runs the loaded garden at **30.0 ticks/s with the GPU at ~81 %**. The final EDRAM contents are bit-identical
to the original shader. One honest caveat: about 100 pixels of one intermediate pass differ by ±1 in an 8-bit color
channel (see [Exactness](#exactness)).

Everything is behind one setting, `edram_interlock_optimizations` (default off), plus `readback_resolve = "lazy"`.
Research environment variables (`REX_ROV_*`, `REX_VK_*`) switch the parts separately.

## Results

RX 6600 (RDNA2, Adrenalin 26.8.1), Windows 11, Viva Piñata (4D5307F2), 1280×720, resolution scale 1.

| Configuration | Frozen frame (tick 1200), frames/s | GPU ms/frame | Live garden, ticks/s |
| :-- | --: | --: | --: |
| D3D12 host render targets (RTV, not exact) | — | — | 30 (GPU ~60 %) |
| D3D12 ROV, original + full readback | — | 80 (title, vsync off) | 13–15 |
| Vulkan FSI, original shader + lazy readback | 8.4 | 118 | — |
| Vulkan FSI, all changes below + lazy readback | 35.0 | 28 | **30.0** (GPU ~81 %) |

"Frozen frame": a debug hook stops the game world at tick 1200 and the same frame keeps being drawn with vsync off,
so runs are comparable. GPU ms/frame = GPU utilization × 10 / fps (Windows GPU engine counters).

## What was slow

Radeon GPU Profiler on the terrain pixel shaders under D3D12 ROV: **83–92 % of the shader time was the driver's POPS
wait loop** (`s_sleep` polling until the overlapping earlier waves leave the ordered section); the real work was under
12 %. POPS (primitive ordered pixel shading, how RDNA implements ROV / FSI) orders waves, not pixels, so every wave
waits for the whole ordered section of every earlier overlapping wave. The time per frame grows with the **length of
the ordered section** — padding it by ~4k clocks cost +40 % GPU time.

On the CPU side, `readback_resolve = "full"` (the exact readback of resolves for the CPU) waited for the GPU about 25
times per frame: the command processor thread was blocked 67–80 % of the time.

So the plan: make the ordered section short, enter it less often, and stop the CPU from waiting.

## What we changed

Each change keeps the EDRAM operations and their order; only where the work is done changes.

1. **Late depth / stencil test** (`REX_ROV_LATE_DEPTH`). The test runs at the end of the shader inside a short ordered
   section instead of wrapping the whole shader. Exact by construction for shaders that don't kill pixels or write
   depth (the only ones the original runs early).
2. **Known render target formats** (`REX_ROV_KNOWN_FORMATS`, "M2"). The formats of RT 0/1 and frequent RT 0 blend
   modes go into the pixel shader modification, so the driver folds the format and blending branches. D3D12 ROV,
   title garden, vsync off: 80.3 ms GPU per tick originally, 50.9 ms with (1), 37.7 ms with (1) + (2).
3. **Short ordered section with batched loads** (`REX_VK_FSI_SHORT`, `REX_VK_FSI_BATCH`, "M3"). Everything that
   doesn't touch the EDRAM (depth per sample, color clamping and packing) is done before
   `OpBeginInvocationInterlockEXT`; inside, the EDRAM loads of all samples are issued at once.
4. **Exact early depth rejection** (`REX_VK_FSI_EARLY_REJECT`, "M4"). Between two EDRAM barriers ("epoch") the render
   target cache guarantees that a depth dword is written only by the depth test of the same depth buffer. If every
   draw of the epoch writes depth only with LESS / LEQUAL / EQUAL (or the mirror GREATER / GEQUAL), the stored depth is
   monotonic, so a stale read before the ordered section can only be *less* strict than the real value: a fragment
   failing against it fails against the real one too, and with stencil off a failing fragment writes nothing. The
   command processor tracks the epoch per draw (`UpdateFsiEarlyDepthReject`, also poisoned when a color target of the
   draw may overlap the depth range) and allows the rejection with a system flag. VP1: allowed for 97 % of the draws,
   −13 % GPU time.
5. **Lazy readback** (`readback_resolve = "lazy"`, Vulkan). A resolve the CPU might read is copied to the guest memory
   only when the CPU touches those pages: the pages are protected (`PhysicalHeap` data providers), the access fault
   waits for that GPU submission (timeline semaphore, any thread) and copies the data. The CPU sees exactly the data
   `full` gives, at the same point of the command stream, without stalling the GPU. VP1 almost never reads resolves
   (0–8 provider calls per 5 s).
6. **AMD driver bug workaround.** `vkWaitForFences` with more than ~64 fences returns `VK_ERROR_UNKNOWN` on the AMD
   Windows driver (seen with 89 fences in flight during save-and-quit, ~1500 draws dropped). Waits go in chunks of 32.

## Exactness

The check is built into the plugin: `REX_VK_FSI_AB=<dir>` plus a trigger file `go` on a frozen frame dumps the EDRAM
and every GPU-written guest range of the frame with the reference variant (a1), the research variant (b) and the
reference again (a2). a1 = a2 shows that the frame is frozen, a1 = b that the variant is exact. The reference is the
**original shader with full readback**. Negative controls make sure the check sees errors: corrupting the early
rejection (`REX_VK_FSI_EARLY_REJECT_SABOTAGE=1`) gives ~28 000 different bytes, a deliberately shifted lazy copy
(`REX_VK_AB_READBACK_SABOTAGE`) is caught too.

Results on VP1 (attract garden and a loaded save, 2× MSAA main scene):

- (3) + (4) + (5) against (1) + (2): **0 differences** in the 10 MB EDRAM and in 66 MB of GPU-written guest memory.
- Everything against the original shader: **EDRAM identical**; guest memory **997 of 66 MB bytes differ by ±1** in
  one color channel, about 100 pixels in one area of the 2× MSAA main scene and the bloom downsampling chain that is
  made from it. Depth resolves are identical. The difference repeats at the same addresses run after run (a1 = a2), so
  it is not a race.
- Known formats alone with the early test: 5 pixels ±1.

**Hypothesis (not proven):** the driver picks different floating-point instruction sequences for differently shaped
shaders, so the same math rounds differently in rare cases. Taking the depth derivatives at the beginning of the
shader (where the original early test takes them) changed nothing.

**Open question:** the late test *with unknown formats* (a research-only combination) is not deterministic from frame
to frame (a1 ≠ a2 by ~5 000 bytes in the same MSAA pass). We haven't found why; the shipped combination is stable.

## Limits

- Tested on one GPU (RX 6600, RDNA2) and one game. NVIDIA and Intel implement interlock differently; it may be faster
  or slower there, and the exactness check should be repeated.
- **Hypothesis:** the AMD driver hoists the POPS wait to the whole wave, so rejected lanes still wait — (4) shortens
  sections but can't skip the wait. Seen in the ISA, not measured.
- D3D12 got (1) and (2) only. (3)–(5) should port to DXBC / D3D12 with the same gains (not done yet).
- Higher resolution scales cost more: the FSI cost grows with the pixel count.

## How to use / reproduce

- Game settings: `graphics_backend = "vulkan"`, `render_target_path_vulkan = "fsi"`, `readback_resolve = "lazy"`,
  `edram_interlock_optimizations = true`. (In VivaPinataRecomp this is the launcher's *Accurate* mode.)
- Parts: `REX_ROV_LATE_DEPTH`, `REX_ROV_KNOWN_FORMATS`, `REX_VK_FSI_SHORT`, `REX_VK_FSI_BATCH`,
  `REX_VK_FSI_EARLY_REJECT` (`0`/`1`; `2` = the unconditional, inexact early rejection for comparison). An unset
  variable follows the setting.
- Diagnostics: `REX_VK_SYNC_STATS=1` (CPU–GPU waits), `REX_VK_DEPTH_STATS=1` (depth functions, epochs, rejection
  share), `REX_VK_DUMP_ISA` (fragment shader ISA via `VK_AMD_shader_info`), `REX_VK_AB_LOG_RESOLVES=1` (the source of
  every resolve).
- The A/B check: `REX_VK_FSI_AB=<dir>`, create `<dir>/go` on a frozen frame, read `<dir>/report.txt`;
  `REX_VK_AB_READBACK=1` also compares full against lazy readback.

The commits on `rov-research` are research history (switches, probes, then the exact versions); they will be
reorganized into clean commits if this goes upstream.
