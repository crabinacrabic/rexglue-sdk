#pragma once
/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2015 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 *
 * @modified    Tom Clay, 2026 - Adapted for ReXGlue runtime
 */

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <iterator>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include <rex/graphics/format/ucode.h>
#include <rex/graphics/pipeline/shader/shader.h>
#include <rex/graphics/registers.h>
#include <rex/graphics/xenos.h>
#include <rex/math.h>
#include <rex/string/buffer.h>

namespace rex::graphics {

// ROV research: REX_ROV_LATE_DEPTH=1 forces the late (end of shader) EDRAM
// depth / stencil test in the ROV and FSI paths, so the ordered section only
// covers the tail of the pixel shader.
inline bool RovLateDepthDebug() {
  static const bool value = [] {
    const char* env = std::getenv("REX_ROV_LATE_DEPTH");
    return env && *env == '1';
  }();
  return value;
}

// ROV research: REX_ROV_SAMPLE_RATE=1 runs the D3D12 ROV pixel shader at sample
// frequency (SV_SampleIndex input, coverage masked to the own sample), so POPS
// orders overlapping invocations per sample instead of per pixel. Formally not
// allowed with ForcedSampleCount (UAV-only rendering) - an experiment.
inline bool RovSampleRateDebug() {
  static const bool value = [] {
    const char* env = std::getenv("REX_ROV_SAMPLE_RATE");
    return env && *env == '1';
  }();
  return value;
}

// ROV research: REX_ROV_HOIST=1 prepares the color (exponent bias, clamping,
// packing without blending, the EDRAM address) before the late depth / stencil
// test, so the ordered section only contains the EDRAM read-modify-write.
// ROV research: REX_ROV_PAD=N adds N xorshift steps (3 dependent integer ops
// each) inside the ordered section, right after the late depth / stencil test,
// to measure how the frame time depends on the length of the ordered section.
// The result only discards a pixel if it equals one 32-bit constant.
inline uint32_t RovPadDebug() {
  static const uint32_t value = [] {
    const char* env = std::getenv("REX_ROV_PAD");
    return env ? uint32_t(std::strtoul(env, nullptr, 10)) : uint32_t(0);
  }();
  return value;
}

// ROV research: REX_ROV_KNOWN_FORMATS=1 passes the color render target formats
// in the pixel shader modification, so the ROV EDRAM code uses literal format
// flags instead of the system constant and the driver folds the format switches.
inline bool RovKnownFormatsDebug() {
  static const bool value = [] {
    const char* env = std::getenv("REX_ROV_KNOWN_FORMATS");
    return env && *env == '1';
  }();
  return value;
}

// ROV research (REX_ROV_KNOWN_FORMATS): frequent RB_BLENDCONTROL & 0x1FFF1FFF
// values (edram_rt_blend_factors_ops) that the pixel shader modification can
// carry as index + 1, so the ROV / FSI code uses them as literals.
inline constexpr uint32_t kRovKnownBlendFactorsOps[7] = {
    0x00010001,  // No blending.
    0x01010001,  // Color = source, alpha = source + destination.
    0x01010706,  // Color alpha blending, alpha = source + destination.
    0x07060706,  // Alpha blending of both.
    0x01010101,  // Additive.
    0x01060106,  // Source alpha weighted additive.
    0x07010706,  // Color alpha blending, alpha = source + (1 - sa) * destination.
};
// Index + 1 of the value in kRovKnownBlendFactorsOps, 0 if it's not there.
inline uint32_t RovKnownBlendIndex(uint32_t blend_factors_ops) {
  for (uint32_t i = 0; i < uint32_t(std::size(kRovKnownBlendFactorsOps)); ++i) {
    if (kRovKnownBlendFactorsOps[i] == blend_factors_ops) {
      return i + 1;
    }
  }
  return 0;
}

inline bool RovHoistDebug() {
  static const bool value = [] {
    const char* env = std::getenv("REX_ROV_HOIST");
    return env && *env == '1';
  }();
  return value;
}

// ROV research (REX_VK_FSI_AB): the variant forced by the in-run exactness
// check: -1 when not forcing, 0 for the reference (the REX_VK_FSI_* research
// switches off), 1 for the research variant (the switches as set).
inline std::atomic<int>& RovAbVariant() {
  static std::atomic<int> variant{-1};
  return variant;
}

// ROV research: REX_VK_FSI_SHORT=1 (with REX_ROV_LATE_DEPTH=1) builds the
// Vulkan FSI epilogue with a short ordered section - everything that doesn't
// read the EDRAM is done before OpBeginInvocationInterlockEXT.
inline bool RovShortSectionDebug() {
  static const bool env_value = [] {
    const char* env = std::getenv("REX_VK_FSI_SHORT");
    return env && *env == '1';
  }();
  return RovLateDepthDebug() && env_value &&
         RovAbVariant().load(std::memory_order_relaxed) != 0;
}

// ROV research: REX_VK_FSI_BATCH=1 (with REX_VK_FSI_SHORT=1) issues the EDRAM
// loads of all samples (depth / stencil and render target 0) at the beginning
// of the ordered section, so it waits for the memory once, not per sample.
inline bool RovBatchLoadsDebug() {
  static const bool env_value = [] {
    const char* env = std::getenv("REX_VK_FSI_BATCH");
    return env && *env == '1';
  }();
  return RovShortSectionDebug() && env_value;
}

// ROV research, NOT EXACT - only to measure the potential of a conservative
// early depth rejection (M4): REX_VK_FSI_EARLY_REJECT=1 (with
// REX_VK_FSI_SHORT=1) reads the depth before the ordered section (unordered)
// and skips the section if no covered sample passes the depth test against it
// (only without stencil).
inline bool RovEarlyRejectDebug() {
  static const bool env_value = [] {
    const char* env = std::getenv("REX_VK_FSI_EARLY_REJECT");
    return env && *env == '1';
  }();
  return RovShortSectionDebug() && env_value;
}

class ShaderTranslator {
 public:
  virtual ~ShaderTranslator();

  virtual uint64_t GetDefaultVertexShaderModification(
      uint32_t /*dynamic_addressable_register_count*/,
      Shader::HostVertexShaderType /*host_vertex_shader_type*/ =
          Shader::HostVertexShaderType::kVertex) const {
    return 0;
  }
  virtual uint64_t GetDefaultPixelShaderModification(
      uint32_t /*dynamic_addressable_register_count*/) const {
    return 0;
  }

  // AnalyzeUcode must be done on the shader before translating!
  bool TranslateAnalyzedShader(Shader::Translation& translation);

 protected:
  ShaderTranslator();

  // Resets translator state before beginning translation.
  virtual void Reset();

  // Shader and modification currently being translated.
  Shader::Translation& current_translation() const { return *translation_; }
  Shader& current_shader() const { return current_translation().shader(); }

  // Register count from SQ_PROGRAM_CNTL, stored by the implementation in its
  // modification bits.
  virtual uint32_t GetModificationRegisterCount() const { return xenos::kMaxShaderTempRegisters; }

  // True if the current shader is a vertex shader.
  bool is_vertex_shader() const { return current_shader().type() == xenos::ShaderType::kVertex; }
  // True if the current shader is a pixel shader.
  bool is_pixel_shader() const { return current_shader().type() == xenos::ShaderType::kPixel; }

  // Temporary register count, accessible via static and dynamic addressing.
  uint32_t register_count() const { return register_count_; }

  // Emits a translation error that will be passed back in the result.
  virtual void EmitTranslationError(const char* message, bool is_fatal = true);

  // Handles the start of translation.
  // At this point the vertex and texture bindings have been gathered.
  virtual void StartTranslation() {}

  // Handles the end of translation when all ucode has been processed.
  // Returns the translated shader binary.
  virtual std::vector<uint8_t> CompleteTranslation() { return std::vector<uint8_t>(); }

  // Handles post-translation tasks when the shader has been fully translated.
  virtual void PostTranslation() {}
  // Sets the host disassembly on a shader.
  void set_host_disassembly(Shader::Translation& translation, std::string value) {
    translation.host_disassembly_ = std::move(value);
  }

  // Pre-process a control-flow instruction before anything else.
  virtual void PreProcessControlFlowInstructions(
      std::vector<ucode::ControlFlowInstruction> /*instrs*/) {}

  // Handles translation for control flow label addresses.
  // This is triggered once for each label required (due to control flow
  // operations) before any of the instructions within the target exec.
  virtual void ProcessLabel(uint32_t /*cf_index*/) {}

  // Handles translation for control flow nop instructions.
  virtual void ProcessControlFlowNopInstruction(uint32_t /*cf_index*/) {}
  // Handles the start of a control flow instruction at the given address.
  virtual void ProcessControlFlowInstructionBegin(uint32_t /*cf_index*/) {}
  // Handles the end of a control flow instruction that began at the given
  // address.
  virtual void ProcessControlFlowInstructionEnd(uint32_t /*cf_index*/) {}
  // Handles translation for control flow exec instructions prior to their
  // contained ALU/fetch instructions.
  virtual void ProcessExecInstructionBegin(const ParsedExecInstruction& /*instr*/) {}
  // Handles translation for control flow exec instructions after their
  // contained ALU/fetch instructions.
  virtual void ProcessExecInstructionEnd(const ParsedExecInstruction& /*instr*/) {}
  // Handles translation for loop start instructions.
  virtual void ProcessLoopStartInstruction(const ParsedLoopStartInstruction& /*instr*/) {}
  // Handles translation for loop end instructions.
  virtual void ProcessLoopEndInstruction(const ParsedLoopEndInstruction& /*instr*/) {}
  // Handles translation for function call instructions.
  virtual void ProcessCallInstruction(const ParsedCallInstruction& /*instr*/) {}
  // Handles translation for function return instructions.
  virtual void ProcessReturnInstruction(const ParsedReturnInstruction& /*instr*/) {}
  // Handles translation for jump instructions.
  virtual void ProcessJumpInstruction(const ParsedJumpInstruction& /*instr*/) {}
  // Handles translation for alloc instructions. Memory exports for eM#
  // indicated by export_eM must be performed, regardless of the alloc type.
  virtual void ProcessAllocInstruction(const ParsedAllocInstruction& /*instr*/,
                                       uint8_t /*export_eM*/) {}

  // Handles translation for vertex fetch instructions.
  virtual void ProcessVertexFetchInstruction(const ParsedVertexFetchInstruction& /*instr*/) {}
  // Handles translation for texture fetch instructions.
  virtual void ProcessTextureFetchInstruction(const ParsedTextureFetchInstruction& /*instr*/) {}
  // Handles translation for ALU instructions.
  // memexport_eM_potentially_written_before needs to be handled by `kill`
  // instruction to make sure memory exports for the eM# writes earlier in
  // previous execs and the current exec are done before the invocation becomes
  // inactive.
  virtual void ProcessAluInstruction(const ParsedAluInstruction& /*instr*/,
                                     uint8_t /*memexport_eM_potentially_written_before*/) {}

 private:
  void TranslateControlFlowInstruction(const ucode::ControlFlowInstruction& cf);
  void TranslateExecInstructions(const ParsedExecInstruction& instr);

  // Current shader and modification being translated.
  Shader::Translation* translation_ = nullptr;

  // Accumulated translation errors.
  std::vector<Shader::Error> errors_;

  // Temporary register count, accessible via static and dynamic addressing.
  uint32_t register_count_ = 0;

  // Current control flow dword index.
  uint32_t cf_index_ = 0;

  // Kept for supporting vfetch_mini.
  ucode::VertexFetchInstruction previous_vfetch_full_;
};

}  // namespace rex::graphics
