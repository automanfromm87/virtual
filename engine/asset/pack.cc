#include "engine/asset/pack.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <fstream>
#include <iterator>

#include "engine/shaders/shader_types.h"

namespace eng::asset {
namespace {

constexpr char kMagic[8] = {'V', 'I', 'R', 'T', 'P', 'A', 'K', '1'};
constexpr std::uint32_t kVersion = 1;

// Everything in the file is little-endian and naturally aligned, which is what
// lets the reader point at it instead of copying. Every field is a fixed width:
// `int` and `size_t` are the two types that would make a package written on one
// machine unreadable on another.
struct Header {
    char magic[8];
    std::uint32_t version;
    std::uint32_t entry_count;
    std::uint64_t names_offset;
    std::uint64_t names_size;
    std::uint64_t data_offset;
    std::uint64_t data_size;
    std::uint32_t table_crc;  // over the entry table and the name blob
    std::uint32_t reserved;
};
static_assert(sizeof(Header) == 56, "the package header must not drift");

// FileEntry, not Entry: PackWriter has its own Entry for the un-serialised
// form, and inside a PackWriter member the class's nested name wins -- so a
// bare `Entry` there silently means the wrong struct.
struct FileEntry {
    std::uint64_t name_offset;
    std::uint32_t name_length;
    std::uint32_t type;
    std::uint64_t data_offset;
    std::uint64_t data_size;
    std::uint32_t crc;
    std::uint32_t reserved;
};
static_assert(sizeof(FileEntry) == 40, "the package entry must not drift");

// The cooked mesh's own little header, at the start of a Mesh entry's payload.
struct MeshHeader {
    std::uint32_t vertex_count;
    std::uint32_t index_count;
    std::uint32_t vertex_stride;
    std::uint32_t index_stride;
    float centre[3];
    float radius;
};
static_assert(sizeof(MeshHeader) == 32, "the cooked mesh header must not drift");

// ALIGNED to sixteen. A vertex block handed straight to the GPU has to be
// aligned for the loads the vertex stage does, and a package whose second mesh
// starts on an odd byte works on Apple silicon and faults elsewhere.
std::size_t Align16(std::size_t n) { return (n + 15u) & ~std::size_t(15u); }

}  // namespace

std::uint32_t Crc32(std::span<const std::uint8_t> data) {
    // The table is built once, on first use. A bit-at-a-time implementation is
    // eight times slower and this runs over every byte of every package.
    static const std::array<std::uint32_t, 256> table = [] {
        std::array<std::uint32_t, 256> t{};
        for (std::uint32_t i = 0; i < 256; ++i) {
            std::uint32_t c = i;
            for (int k = 0; k < 8; ++k)
                c = (c & 1u) ? (0xEDB88320u ^ (c >> 1)) : (c >> 1);
            t[i] = c;
        }
        return t;
    }();
    std::uint32_t crc = 0xFFFFFFFFu;
    for (std::uint8_t b : data) crc = table[(crc ^ b) & 0xFFu] ^ (crc >> 8);
    return crc ^ 0xFFFFFFFFu;
}

// ------------------------------------------------------------------- writing

void PackWriter::Add(std::string_view name, AssetType type,
                     std::span<const std::uint8_t> data) {
    if (name.empty()) return;
    // A DUPLICATE REPLACES rather than appends. A cooker re-run over a changed
    // source is the normal way to produce one, and a package with two assets
    // of the same name behaves differently depending on which the reader finds
    // first.
    for (auto& e : entries_)
        if (e.name == name) {
            e.type = type;
            e.data.assign(data.begin(), data.end());
            return;
        }
    entries_.push_back({std::string(name), type, {data.begin(), data.end()}});
}

void PackWriter::AddMesh(std::string_view name, const Mesh& mesh) {
    MeshHeader h{};
    h.vertex_count = std::uint32_t(mesh.vertices.size());
    h.index_count = std::uint32_t(mesh.indices.size());
    h.vertex_stride = std::uint32_t(sizeof(VertexIn));
    h.index_stride = std::uint32_t(sizeof(std::uint16_t));
    h.centre[0] = mesh.bounds.center.x;
    h.centre[1] = mesh.bounds.center.y;
    h.centre[2] = mesh.bounds.center.z;
    h.radius = mesh.bounds.radius;

    const std::size_t vertex_bytes = mesh.vertices.size() * sizeof(VertexIn);
    const std::size_t index_bytes = mesh.indices.size() * sizeof(std::uint16_t);
    // The vertex block starts on a sixteen-byte boundary WITHIN the payload, so
    // that an aligned payload gives an aligned vertex block.
    const std::size_t vertex_at = Align16(sizeof(MeshHeader));
    const std::size_t index_at = Align16(vertex_at + vertex_bytes);

    std::vector<std::uint8_t> payload(index_at + index_bytes, 0);
    std::memcpy(payload.data(), &h, sizeof(h));
    if (vertex_bytes > 0)
        std::memcpy(payload.data() + vertex_at, mesh.vertices.data(), vertex_bytes);
    if (index_bytes > 0)
        std::memcpy(payload.data() + index_at, mesh.indices.data(), index_bytes);
    Add(name, AssetType::Mesh, payload);
}

std::vector<std::uint8_t> PackWriter::Build() const {
    std::vector<std::uint8_t> names;
    std::vector<FileEntry> table(entries_.size());
    std::size_t data_size = 0;
    for (std::size_t i = 0; i < entries_.size(); ++i) {
        table[i].name_offset = names.size();
        table[i].name_length = std::uint32_t(entries_[i].name.size());
        names.insert(names.end(), entries_[i].name.begin(), entries_[i].name.end());
        table[i].type = std::uint32_t(entries_[i].type);
        table[i].data_offset = data_size;
        table[i].data_size = entries_[i].data.size();
        table[i].crc = Crc32(entries_[i].data);
        table[i].reserved = 0;
        data_size = Align16(data_size + entries_[i].data.size());
    }

    const std::size_t table_bytes = table.size() * sizeof(FileEntry);
    Header header{};
    std::memcpy(header.magic, kMagic, 8);
    header.version = kVersion;
    header.entry_count = std::uint32_t(table.size());
    header.names_offset = Align16(sizeof(Header) + table_bytes);
    header.names_size = names.size();
    header.data_offset = Align16(header.names_offset + names.size());
    header.data_size = data_size;
    header.reserved = 0;

    std::vector<std::uint8_t> out(header.data_offset + data_size, 0);
    if (!table.empty())
        std::memcpy(out.data() + sizeof(Header), table.data(), table_bytes);
    if (!names.empty())
        std::memcpy(out.data() + header.names_offset, names.data(), names.size());
    for (std::size_t i = 0; i < entries_.size(); ++i)
        if (!entries_[i].data.empty())
            std::memcpy(out.data() + header.data_offset + table[i].data_offset,
                        entries_[i].data.data(), entries_[i].data.size());

    // The table's checksum is computed over the FINISHED buffer, after the
    // table and names are in place, so it covers exactly the bytes a reader
    // will check.
    header.table_crc =
        Crc32({out.data() + sizeof(Header),
               header.names_offset + names.size() - sizeof(Header)});
    std::memcpy(out.data(), &header, sizeof(header));
    return out;
}

bool PackWriter::WriteFile(const std::string& path, std::string& error) const {
    const std::vector<std::uint8_t> bytes = Build();
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    if (!f) {
        error = "pack: cannot open " + path + " for writing";
        return false;
    }
    f.write(reinterpret_cast<const char*>(bytes.data()),
            std::streamsize(bytes.size()));
    if (!f) {
        error = "pack: writing " + path + " failed part way through";
        return false;
    }
    return true;
}

// ------------------------------------------------------------------- reading

struct Pack::Impl {
    std::vector<std::uint8_t> bytes;
    Header header{};
    std::vector<FileEntry> entries;
    bool valid = false;
};

Pack::Pack() : impl_(std::make_unique<Impl>()) {}
Pack::~Pack() = default;
Pack::Pack(Pack&&) noexcept = default;
Pack& Pack::operator=(Pack&&) noexcept = default;

Pack Pack::Open(std::vector<std::uint8_t> bytes, std::string& error) {
    Pack p;
    Impl& im = *p.impl_;
    error.clear();
    if (bytes.size() < sizeof(Header)) {
        error = "pack: too small to contain a header";
        return p;
    }
    std::memcpy(&im.header, bytes.data(), sizeof(Header));
    if (std::memcmp(im.header.magic, kMagic, 8) != 0) {
        error = "pack: not a package (bad magic)";
        return p;
    }
    if (im.header.version != kVersion) {
        error = "pack: version " + std::to_string(im.header.version) +
                ", expected " + std::to_string(kVersion);
        return p;
    }

    // EVERY OFFSET CHECKED against the actual size before anything is read
    // through it. A package is a file on disk: it can be truncated, it can be
    // from a different build, and it can be corrupt. Trusting its own offsets
    // and then indexing with them is how a bad file becomes an out-of-bounds
    // read instead of an error message.
    const std::size_t table_bytes =
        std::size_t(im.header.entry_count) * sizeof(FileEntry);
    if (sizeof(Header) + table_bytes > bytes.size() ||
        im.header.names_offset + im.header.names_size > bytes.size() ||
        im.header.data_offset + im.header.data_size > bytes.size()) {
        error = "pack: truncated (a section runs past the end of the file)";
        return p;
    }

    const std::uint32_t crc =
        Crc32({bytes.data() + sizeof(Header),
               im.header.names_offset + im.header.names_size - sizeof(Header)});
    if (crc != im.header.table_crc) {
        error = "pack: the entry table's checksum does not match";
        return p;
    }

    im.entries.resize(im.header.entry_count);
    if (im.header.entry_count > 0)
        std::memcpy(im.entries.data(), bytes.data() + sizeof(Header), table_bytes);
    for (const FileEntry& e : im.entries) {
        if (e.name_offset + e.name_length > im.header.names_size ||
            e.data_offset + e.data_size > im.header.data_size) {
            error = "pack: an entry points outside the file";
            im.entries.clear();
            return p;
        }
    }

    im.bytes = std::move(bytes);
    im.valid = true;
    return p;
}

Pack Pack::OpenFile(const std::string& path, std::string& error) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        Pack p;
        error = "pack: cannot open " + path;
        return p;
    }
    std::vector<std::uint8_t> bytes{std::istreambuf_iterator<char>(f),
                                    std::istreambuf_iterator<char>()};
    return Open(std::move(bytes), error);
}

bool Pack::Valid() const { return impl_->valid; }
int Pack::Count() const { return int(impl_->entries.size()); }

std::string_view Pack::Name(int index) const {
    if (index < 0 || std::size_t(index) >= impl_->entries.size()) return {};
    const FileEntry& e = impl_->entries[std::size_t(index)];
    return std::string_view(
        reinterpret_cast<const char*>(impl_->bytes.data() + impl_->header.names_offset +
                                      e.name_offset),
        e.name_length);
}

AssetType Pack::Type(int index) const {
    if (index < 0 || std::size_t(index) >= impl_->entries.size())
        return AssetType::Unknown;
    return AssetType(impl_->entries[std::size_t(index)].type);
}

std::span<const std::uint8_t> Pack::Data(int index) const {
    if (index < 0 || std::size_t(index) >= impl_->entries.size()) return {};
    const FileEntry& e = impl_->entries[std::size_t(index)];
    return {impl_->bytes.data() + impl_->header.data_offset + e.data_offset,
            std::size_t(e.data_size)};
}

int Pack::Find(std::string_view name) const {
    for (int i = 0; i < Count(); ++i)
        if (Name(i) == name) return i;
    return -1;
}

std::span<const std::uint8_t> Pack::Get(std::string_view name) const {
    const int i = Find(name);
    return i >= 0 ? Data(i) : std::span<const std::uint8_t>{};
}

bool Pack::GetMesh(std::string_view name, CookedMesh* out) const {
    if (!out) return false;
    const int index = Find(name);
    if (index < 0 || Type(index) != AssetType::Mesh) return false;
    const std::span<const std::uint8_t> payload = Data(index);
    if (payload.size() < sizeof(MeshHeader)) return false;

    MeshHeader h{};
    std::memcpy(&h, payload.data(), sizeof(h));
    // THE STRIDES ARE CHECKED, not assumed. A package cooked by a build whose
    // VertexIn had a different size would otherwise be read as the current
    // layout -- and a mesh reinterpreted at the wrong stride is not a crash,
    // it is a shape nobody recognises, which reads as a broken exporter.
    if (h.vertex_stride != sizeof(VertexIn) ||
        h.index_stride != sizeof(std::uint16_t))
        return false;

    const std::size_t vertex_at = Align16(sizeof(MeshHeader));
    const std::size_t vertex_bytes = std::size_t(h.vertex_count) * h.vertex_stride;
    const std::size_t index_at = Align16(vertex_at + vertex_bytes);
    const std::size_t index_bytes = std::size_t(h.index_count) * h.index_stride;
    if (index_at + index_bytes > payload.size()) return false;

    out->vertices = payload.subspan(vertex_at, vertex_bytes);
    out->indices = payload.subspan(index_at, index_bytes);
    out->vertex_count = h.vertex_count;
    out->index_count = h.index_count;
    out->bounds.center = Vec3{h.centre[0], h.centre[1], h.centre[2]};
    out->bounds.radius = h.radius;
    return true;
}

std::size_t Pack::PayloadBytes() const {
    std::size_t total = 0;
    for (const FileEntry& e : impl_->entries) total += std::size_t(e.data_size);
    return total;
}

}  // namespace eng::asset
