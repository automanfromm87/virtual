// Screen-sized render targets, by name, that follow the window.
//
// WHY this is a type and not four lines in every app: the code it replaces is
// "if the size changed, destroy the old textures and make new ones", and every
// app wrote its own. Two failure modes, both quiet:
//
//   * Forget to destroy, and the app leaks a full-screen texture per resize.
//   * Forget the size check, and it reallocates every frame — which works, and
//     just gets slower and stutters under memory pressure.
//
// Neither shows up as a wrong picture, so neither gets noticed. Allocations()
// exists so a test can assert the second one directly.
#pragma once

#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "engine/rhi/rhi.h"

namespace eng::app {

class FrameTargets {
  public:
    FrameTargets(rhi::Device& device, rhi::Format color);
    ~FrameTargets();

    FrameTargets(const FrameTargets&) = delete;
    FrameTargets& operator=(const FrameTargets&) = delete;

    // Once per frame, before any target is asked for. A no-op when the size is
    // unchanged, which is the overwhelmingly common case.
    void Resize(int width, int height);

    // Created on first use at the current size, and recreated after a Resize.
    // The returned handle is only valid until the next Resize that actually
    // changes something — hold names, not handles.
    [[nodiscard]] rhi::TextureId Color(std::string_view name);
    // A HALF-FLOAT target, for anything a later pass needs the real brightness
    // of. A scene target has to be one of these now that the tone map lives in
    // the composite. `divisor` shrinks it: bloom works at half or quarter
    // resolution, which is both cheaper and a wider blur for the same kernel.
    [[nodiscard]] rhi::TextureId Hdr(std::string_view name, int divisor = 1);
    // `sampleable` costs real memory: an ordinary depth target is memoryless
    // and cannot be read by a later pass, which is what SSAO needs.
    [[nodiscard]] rhi::TextureId Depth(std::string_view name,
                                       bool sampleable = false);

    [[nodiscard]] int Width() const { return width_; }
    [[nodiscard]] int Height() const { return height_; }

    // Total textures ever created. A steady-state frame must not increase it.
    [[nodiscard]] int Allocations() const { return allocations_; }
    [[nodiscard]] int Live() const { return int(targets_.size()); }

  private:
    struct Target {
        rhi::TextureId id;
        bool is_depth = false;
        bool sampleable = false;
    };
    Target& Lookup(std::string_view name, bool is_depth, bool sampleable,
                   rhi::Format format, int divisor);
    void DestroyAll();

    rhi::Device& device_;
    rhi::Format color_;
    int width_ = 0, height_ = 0;
    int allocations_ = 0;
    std::unordered_map<std::string, Target> targets_;
};

}  // namespace eng::app
