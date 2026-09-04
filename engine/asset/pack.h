// A baked asset package: one file holding many assets in the form the engine
// actually uses them.
//
// WHY NOT KEEP LOADING glTF AND PNG. Because those are INTERCHANGE formats,
// designed to be produced by one tool and read by another, and everything that
// makes them good at that makes them bad at load time. A glTF is JSON: parsing
// it means tokenising text, allocating a document, walking it, and converting
// every attribute into the layout the GPU wants. A PNG is compressed with an
// algorithm chosen for size over speed. Both are the right thing to author in
// and the wrong thing to ship.
//
// A baked package is the opposite trade. It is produced once, offline, by a
// cooker that has already done the parsing and the conversion; loading it is a
// read and a bounds check. The vertex data in it is byte-identical to what goes
// into a GPU buffer, so an upload is a memcpy rather than a transformation.
//
// WHAT THIS BUYS, concretely: a level that takes two seconds to load from source
// assets loads in the time it takes to read the file. That is the difference
// between iterating on a game and waiting to iterate on a game, and it is why
// this exists before an editor does -- an editor that takes two seconds to
// reload is one nobody uses.
//
// FORMAT, deliberately simple enough to read in a hex editor:
//
//   [header][entry table][name blob][data blob]
//
// No compression. Compression trades load time for disk, and disk is not the
// constraint -- but an entry's payload is opaque here, so a caller that wants
// its own compression can store compressed bytes and say so in `type`.
#pragma once

#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "engine/core/math.h"
#include "engine/geometry/mesh.h"

namespace eng::asset {

// What an entry holds. The reader does not interpret payloads -- this is for
// the caller to dispatch on, and for a diagnostic listing to be readable.
enum class AssetType : std::uint32_t {
    Unknown = 0,
    Mesh = 1,
    Texture = 2,
    Clip = 3,      // an animation clip
    Scene = 4,     // a serialised scene
    Audio = 5,
    Raw = 6,       // anything else
};

// A cooked mesh: the exact bytes an upload wants, plus what the renderer needs
// to know about them without looking.
//
// The vertex block is an array of VertexIn and the index block an array of
// uint32, both in the engine's one and only layout -- so `CreateBuffer(vertices)`
// is the whole of loading it. A format that stored positions and normals in
// separate streams, or in a different precision, would need a conversion pass
// and would stop being a memcpy.
struct CookedMesh {
    std::span<const std::uint8_t> vertices;
    std::span<const std::uint8_t> indices;
    std::uint32_t vertex_count = 0;
    std::uint32_t index_count = 0;
    Bounds bounds;
};

// --- writing ---------------------------------------------------------------

class PackWriter {
  public:
    // Names must be unique. A duplicate REPLACES the earlier entry rather than
    // appending a second one with the same name: a package with two assets
    // called "rock" is one whose behaviour depends on which the reader finds
    // first, and a cooker re-running over a changed source is the normal way to
    // produce one.
    void Add(std::string_view name, AssetType, std::span<const std::uint8_t> data);
    // Cooks a mesh into the byte layout above and adds it.
    void AddMesh(std::string_view name, const Mesh&);

    [[nodiscard]] int Count() const { return int(entries_.size()); }
    // Serialises everything into one buffer.
    [[nodiscard]] std::vector<std::uint8_t> Build() const;
    [[nodiscard]] bool WriteFile(const std::string& path, std::string& error) const;

  private:
    struct Entry {
        std::string name;
        AssetType type = AssetType::Unknown;
        std::vector<std::uint8_t> data;
    };
    std::vector<Entry> entries_;
};

// --- reading ---------------------------------------------------------------

class Pack {
  public:
    Pack();
    ~Pack();
    Pack(Pack&&) noexcept;
    Pack& operator=(Pack&&) noexcept;
    Pack(const Pack&) = delete;
    Pack& operator=(const Pack&) = delete;

    // OWNS the bytes. A version that borrowed them would be faster and would
    // make every caller responsible for keeping a buffer alive exactly as long
    // as the spans it handed out -- which is the kind of lifetime rule that
    // works until someone reloads a package while the old one is still bound.
    [[nodiscard]] static Pack Open(std::vector<std::uint8_t> bytes,
                                   std::string& error);
    [[nodiscard]] static Pack OpenFile(const std::string& path, std::string& error);

    [[nodiscard]] bool Valid() const;
    [[nodiscard]] int Count() const;
    [[nodiscard]] std::string_view Name(int index) const;
    [[nodiscard]] AssetType Type(int index) const;
    [[nodiscard]] std::span<const std::uint8_t> Data(int index) const;

    [[nodiscard]] int Find(std::string_view name) const;
    [[nodiscard]] std::span<const std::uint8_t> Get(std::string_view name) const;
    // Reinterprets an entry as a cooked mesh. Returns false when the entry is
    // not a mesh or its header does not match its size -- which is the check
    // that catches a truncated file, and it has to happen here rather than at
    // the first draw.
    [[nodiscard]] bool GetMesh(std::string_view name, CookedMesh* out) const;

    // Total payload bytes, for a load-time report.
    [[nodiscard]] std::size_t PayloadBytes() const;

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// The checksum the format uses. Exposed because a caller cooking into a package
// may want to verify its own inputs the same way.
//
// CRC-32, not a cryptographic hash. The threat here is a truncated write, a bad
// disk or a version mismatch -- accidents, not adversaries -- and for those a
// CRC is as good and a great deal faster.
[[nodiscard]] std::uint32_t Crc32(std::span<const std::uint8_t>);

}  // namespace eng::asset
