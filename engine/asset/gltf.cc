#include "engine/asset/gltf.h"

#include "engine/asset/png.h"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <sstream>

#include "engine/asset/json.h"

namespace eng::gltf {
namespace {

// glTF componentType codes.
constexpr int kByte = 5120, kUByte = 5121, kShort = 5122, kUShort = 5123;
constexpr int kUInt = 5125, kFloat = 5126;

int ComponentSize(int type) {
    switch (type) {
        case kByte: case kUByte: return 1;
        case kShort: case kUShort: return 2;
        case kUInt: case kFloat: return 4;
        default: return 0;
    }
}

int ComponentCount(const std::string& type) {
    if (type == "SCALAR") return 1;
    if (type == "VEC2") return 2;
    if (type == "VEC3") return 3;
    if (type == "VEC4") return 4;
    if (type == "MAT4") return 16;
    return 0;
}

// Everything needed to walk one accessor's elements out of a buffer.
struct AccessorView {
    const std::uint8_t* base = nullptr;
    std::size_t stride = 0;
    int count = 0;
    int components = 0;
    int component_type = 0;
    bool valid = false;
};

class Reader {
  public:
    Reader(const json::Value& root, const std::vector<std::uint8_t>& bin,
           std::string& error)
        : root_(root), error_(error) {
        // Buffers: a data uri decodes from the JSON, anything else takes the
        // caller-supplied blob.
        const json::Value& buffers = root["buffers"];
        for (std::size_t i = 0; i < buffers.Size(); ++i) {
            const std::string& uri = buffers[i]["uri"].Str();
            const std::string_view kPrefix = "base64,";
            const std::size_t at = uri.find(kPrefix);
            std::vector<std::uint8_t> data;
            if (uri.rfind("data:", 0) == 0 && at != std::string::npos) {
                if (!json::DecodeBase64(std::string_view(uri).substr(at + kPrefix.size()),
                                        data)) {
                    Fail("buffer " + std::to_string(i) + " has invalid base64");
                    return;
                }
            } else {
                data = bin;  // external .bin, supplied by the caller
            }
            buffers_.push_back(std::move(data));
        }
    }

    // Raw bytes of a bufferView, for images stored inside the buffer rather
    // than as their own file. Empty span if the index is out of range.
    [[nodiscard]] std::span<const std::uint8_t> BufferViewBytes(int index) const {
        const json::Value& bv = root_["bufferViews"][std::size_t(index)];
        if (bv.IsNull()) return {};
        const int buf = bv["buffer"].Int(-1);
        if (buf < 0 || std::size_t(buf) >= buffers_.size()) return {};
        const std::size_t off = std::size_t(bv["byteOffset"].Int(0));
        const std::size_t len = std::size_t(bv["byteLength"].Int(0));
        if (off + len > buffers_[std::size_t(buf)].size()) return {};
        return std::span<const std::uint8_t>(buffers_[std::size_t(buf)].data() + off, len);
    }

    [[nodiscard]] bool Failed() const { return !error_.empty(); }
    void Fail(const std::string& what) {
        if (error_.empty()) error_ = "gltf: " + what;
    }

    AccessorView View(int index) {
        AccessorView v;
        const json::Value& acc = root_["accessors"][std::size_t(index)];
        if (acc.IsNull()) { Fail("missing accessor " + std::to_string(index)); return v; }
        if (acc.Has("sparse")) { Fail("sparse accessors are not supported"); return v; }

        v.count = acc["count"].Int();
        v.component_type = acc["componentType"].Int();
        v.components = ComponentCount(acc["type"].Str());
        const int csize = ComponentSize(v.component_type);
        if (v.components == 0 || csize == 0) { Fail("unknown accessor type"); return v; }

        const int bv_index = acc["bufferView"].Int(-1);
        if (bv_index < 0) { Fail("accessor without a bufferView"); return v; }
        const json::Value& bv = root_["bufferViews"][std::size_t(bv_index)];
        const int buf = bv["buffer"].Int();
        if (buf < 0 || std::size_t(buf) >= buffers_.size()) { Fail("bad buffer index"); return v; }

        const std::size_t offset =
            std::size_t(bv["byteOffset"].Int(0)) + std::size_t(acc["byteOffset"].Int(0));
        // byteStride absent means tightly packed. Interleaved vertex data is
        // extremely common in exported files, so this is not optional.
        v.stride = std::size_t(bv["byteStride"].Int(0));
        if (v.stride == 0) v.stride = std::size_t(csize * v.components);

        const std::vector<std::uint8_t>& data = buffers_[std::size_t(buf)];
        const std::size_t need = offset + v.stride * std::size_t(v.count > 0 ? v.count - 1 : 0) +
                                 std::size_t(csize * v.components);
        if (v.count > 0 && need > data.size()) { Fail("accessor runs past the buffer"); return v; }

        v.base = data.data() + offset;
        v.valid = true;
        return v;
    }

    // Reads element `i` as up to four floats, converting from whatever the
    // component type is.
    static void ReadFloats(const AccessorView& v, int i, float* out) {
        const std::uint8_t* p = v.base + v.stride * std::size_t(i);
        for (int c = 0; c < v.components; ++c) {
            switch (v.component_type) {
                case kFloat: {
                    float f;
                    std::memcpy(&f, p + c * 4, 4);
                    out[c] = f;
                    break;
                }
                case kUShort: {
                    std::uint16_t u;
                    std::memcpy(&u, p + c * 2, 2);
                    out[c] = float(u);
                    break;
                }
                case kUByte: out[c] = float(p[c]); break;
                case kShort: {
                    std::int16_t sv;
                    std::memcpy(&sv, p + c * 2, 2);
                    out[c] = float(sv);
                    break;
                }
                case kByte: out[c] = float(std::int8_t(p[c])); break;
                case kUInt: {
                    std::uint32_t u;
                    std::memcpy(&u, p + c * 4, 4);
                    out[c] = float(u);
                    break;
                }
                default: out[c] = 0.0f; break;
            }
        }
    }

    static std::uint32_t ReadIndex(const AccessorView& v, int i) {
        const std::uint8_t* p = v.base + v.stride * std::size_t(i);
        switch (v.component_type) {
            case kUByte: return *p;
            case kUShort: {
                std::uint16_t u;
                std::memcpy(&u, p, 2);
                return u;
            }
            case kUInt: {
                std::uint32_t u;
                std::memcpy(&u, p, 4);
                return u;
            }
            default: return 0;
        }
    }

  private:
    const json::Value& root_;
    std::string& error_;
    std::vector<std::vector<std::uint8_t>> buffers_;
};

Mat4 NodeTransform(const json::Value& n) {
    if (n.Has("matrix")) {
        // glTF stores a matrix COLUMN-major, which is this engine's convention
        // too — so the sixteen floats drop straight in with no transpose.
        const json::Value& m = n["matrix"];
        Mat4 out;
        for (int c = 0; c < 4; ++c) {
            out.col[c] = Vec4{float(m[std::size_t(c * 4 + 0)].Number()),
                              float(m[std::size_t(c * 4 + 1)].Number()),
                              float(m[std::size_t(c * 4 + 2)].Number()),
                              float(m[std::size_t(c * 4 + 3)].Number())};
        }
        return out;
    }
    Mat4 t = Mat4::Identity();
    if (n.Has("translation")) {
        const json::Value& v = n["translation"];
        t = Mat4::Translation(Vec3{float(v[0].Number()), float(v[1].Number()),
                                   float(v[2].Number())});
    }
    Mat4 r = Mat4::Identity();
    if (n.Has("rotation")) {
        const json::Value& v = n["rotation"];
        r = QuatToMat4(Quat{float(v[0].Number()), float(v[1].Number()),
                            float(v[2].Number()), float(v[3].Number())});
    }
    Mat4 s = Mat4::Identity();
    if (n.Has("scale")) {
        const json::Value& v = n["scale"];
        // Mat4::Scale is uniform by design. A non-uniform glTF scale is
        // expressed here as a diagonal matrix, which is correct for positions —
        // the shader's normal transform would need an inverse transpose, and
        // that is a real limitation rather than something to paper over.
        s = Mat4{{{float(v[0].Number()), 0, 0, 0},
                  {0, float(v[1].Number()), 0, 0},
                  {0, 0, float(v[2].Number()), 0},
                  {0, 0, 0, 1}}};
    }
    return t * r * s;  // glTF's defined order
}

void FitBounds(Mesh& m) {
    if (m.vertices.empty()) return;
    Vec3 lo{m.vertices[0].position.x, m.vertices[0].position.y, m.vertices[0].position.z};
    Vec3 hi = lo;
    for (const VertexIn& v : m.vertices) {
        lo.x = std::min(lo.x, v.position.x); hi.x = std::max(hi.x, v.position.x);
        lo.y = std::min(lo.y, v.position.y); hi.y = std::max(hi.y, v.position.y);
        lo.z = std::min(lo.z, v.position.z); hi.z = std::max(hi.z, v.position.z);
    }
    m.bounds.center = (lo + hi) * 0.5f;
    m.bounds.radius = Length(hi - m.bounds.center);
}

}  // namespace

std::vector<Mat4> Document::WorldTransforms() const {
    std::vector<Mat4> world(nodes.size(), Mat4::Identity());
    // Roots first, then children — a glTF node graph is a forest, so one pass
    // over each root's subtree visits every parent before its children.
    std::vector<int> stack(roots.rbegin(), roots.rend());
    while (!stack.empty()) {
        const int i = stack.back();
        stack.pop_back();
        const Node& n = nodes[std::size_t(i)];
        world[std::size_t(i)] =
            n.parent >= 0 ? world[std::size_t(n.parent)] * n.local : n.local;
        for (int c : n.children) stack.push_back(c);
    }
    return world;
}

Document ParseGltf(std::string_view json_text, const std::vector<std::uint8_t>& bin,
                   std::string& error) {
    Document doc;
    error.clear();

    const json::Value root = json::Parse(json_text, error);
    if (!error.empty()) return doc;
    if (!root.IsObject()) { error = "gltf: the top level is not an object"; return doc; }

    // asset.version is a STRING in glTF ("2.0"), not a number — the spec is
    // explicit about it, and reading it as a number silently rejects every
    // valid file.
    const std::string& version = root["asset"]["version"].Str();
    if (version.rfind("2.", 0) != 0) {
        error = "gltf: only version 2 is supported, got \"" + version + "\"";
        return doc;
    }

    Reader reader(root, bin, error);
    if (reader.Failed()) return doc;

    // --- images ---------------------------------------------------------------
    // Decoded before materials so a material can point straight at a Texture2D.
    // A failed decode is recorded as an empty image, not an error: one bad
    // texture should not stop a model from loading.
    {
        const json::Value& images = root["images"];
        for (std::size_t i = 0; i < images.Size(); ++i) {
            const json::Value& im = images[i];
            std::vector<std::uint8_t> bytes;
            std::span<const std::uint8_t> view;

            const std::string& uri = im["uri"].Str();
            const std::string_view kPrefix = "base64,";
            const std::size_t at = uri.find(kPrefix);
            if (uri.rfind("data:", 0) == 0 && at != std::string::npos) {
                if (json::DecodeBase64(std::string_view(uri).substr(at + kPrefix.size()),
                                       bytes))
                    view = bytes;
            } else if (im.Has("bufferView")) {
                view = reader.BufferViewBytes(im["bufferView"].Int(-1));
            } else if (!uri.empty()) {
                // An external file. ParseGltf has no filesystem; LoadGltfFile
                // fills these in afterwards, once it knows the directory.
                doc.images.emplace_back();
                doc.image_uris.push_back(uri);
                continue;
            }

            Texture2D decoded;
            if (!view.empty()) {
                std::string ignored;
                decoded = png::Decode(view, ignored);
            }
            doc.images.push_back(std::move(decoded));
            doc.image_uris.emplace_back();
        }
    }

    // glTF puts a sampler between a material and its image. Nothing above this
    // layer uses the sampler, so collapse the indirection here.
    auto image_of_texture = [&](const json::Value& ref) {
        if (!ref.Has("index")) return -1;
        const int t = ref["index"].Int(-1);
        const json::Value& tex = root["textures"][std::size_t(t)];
        if (tex.IsNull()) return -1;
        const int src = tex["source"].Int(-1);
        return (src >= 0 && std::size_t(src) < doc.images.size()) ? src : -1;
    };

    // --- materials -----------------------------------------------------------
    const json::Value& materials = root["materials"];
    for (std::size_t i = 0; i < materials.Size(); ++i) {
        const json::Value& m = materials[i];
        MaterialDef def;
        def.name = m["name"].Str();
        const json::Value& pbr = m["pbrMetallicRoughness"];
        if (pbr.Has("baseColorFactor")) {
            const json::Value& c = pbr["baseColorFactor"];
            def.base_color = Vec4{float(c[0].Number(1.0)), float(c[1].Number(1.0)),
                                  float(c[2].Number(1.0)), float(c[3].Number(1.0))};
        }
        def.metallic = float(pbr["metallicFactor"].Number(1.0));
        def.roughness = float(pbr["roughnessFactor"].Number(1.0));
        def.base_color_image = image_of_texture(pbr["baseColorTexture"]);
        def.metallic_roughness_image = image_of_texture(pbr["metallicRoughnessTexture"]);
        doc.materials.push_back(def);
    }

    // --- meshes --------------------------------------------------------------
    // glTF mesh -> a list of our primitives, so a node can reference them all.
    std::vector<std::vector<int>> mesh_to_prims;
    const json::Value& meshes = root["meshes"];
    for (std::size_t mi = 0; mi < meshes.Size(); ++mi) {
        std::vector<int> prims;
        const json::Value& prim_list = meshes[mi]["primitives"];
        for (std::size_t pi = 0; pi < prim_list.Size(); ++pi) {
            const json::Value& p = prim_list[pi];
            // mode 4 is TRIANGLES and is the default. Strips and fans would
            // need converting, and no exporter emits them by default.
            if (p["mode"].Int(4) != 4) {
                error = "gltf: only triangle primitives are supported";
                return doc;
            }
            const json::Value& attrs = p["attributes"];
            if (!attrs.Has("POSITION")) {
                error = "gltf: a primitive has no POSITION";
                return doc;
            }

            const AccessorView pos = reader.View(attrs["POSITION"].Int());
            if (!pos.valid) return doc;
            AccessorView nrm, uv;
            const bool has_normal = attrs.Has("NORMAL");
            const bool has_uv = attrs.Has("TEXCOORD_0");
            if (has_normal) { nrm = reader.View(attrs["NORMAL"].Int()); if (!nrm.valid) return doc; }
            if (has_uv) { uv = reader.View(attrs["TEXCOORD_0"].Int()); if (!uv.valid) return doc; }

            if (pos.count > 65535) {
                error = "gltf: primitive has " + std::to_string(pos.count) +
                        " vertices; the engine's index buffers are 16-bit";
                return doc;
            }

            Primitive out;
            out.material = p["material"].Int(-1);
            out.mesh.vertices.reserve(std::size_t(pos.count));
            for (int i = 0; i < pos.count; ++i) {
                float f[4] = {0, 0, 0, 0};
                VertexIn v{};
                Reader::ReadFloats(pos, i, f);
                v.position = Vec4{f[0], f[1], f[2], 0.0f};
                if (has_normal) {
                    float n[4] = {0, 1, 0, 0};
                    Reader::ReadFloats(nrm, i, n);
                    v.normal = Vec4{n[0], n[1], n[2], 0.0f};
                } else {
                    v.normal = Vec4{0, 1, 0, 0};
                }
                if (has_uv) {
                    float t[4] = {0, 0, 0, 0};
                    Reader::ReadFloats(uv, i, t);
                    v.uv = Vec4{t[0], t[1], 0, 0};
                }
                v.color = Vec4{1, 1, 1, 1};
                out.mesh.vertices.push_back(v);
            }

            if (p.Has("indices")) {
                const AccessorView idx = reader.View(p["indices"].Int());
                if (!idx.valid) return doc;
                out.mesh.indices.reserve(std::size_t(idx.count));
                for (int i = 0; i < idx.count; ++i) {
                    const std::uint32_t v = Reader::ReadIndex(idx, i);
                    if (v > 65535) { error = "gltf: index out of 16-bit range"; return doc; }
                    out.mesh.indices.push_back(std::uint16_t(v));
                }
            } else {
                // Non-indexed: synthesise a trivial index buffer so the rest of
                // the engine only ever deals with indexed draws.
                out.mesh.indices.resize(std::size_t(pos.count));
                for (int i = 0; i < pos.count; ++i)
                    out.mesh.indices[std::size_t(i)] = std::uint16_t(i);
            }

            FitBounds(out.mesh);
            prims.push_back(int(doc.primitives.size()));
            doc.primitives.push_back(std::move(out));
        }
        mesh_to_prims.push_back(std::move(prims));
    }

    // --- nodes ---------------------------------------------------------------
    const json::Value& nodes = root["nodes"];
    doc.nodes.resize(nodes.Size());
    for (std::size_t i = 0; i < nodes.Size(); ++i) {
        const json::Value& n = nodes[i];
        Node& out = doc.nodes[i];
        out.name = n["name"].Str();
        out.local = NodeTransform(n);
        const int mesh = n["mesh"].Int(-1);
        if (mesh >= 0 && std::size_t(mesh) < mesh_to_prims.size())
            out.primitives = mesh_to_prims[std::size_t(mesh)];
        const json::Value& kids = n["children"];
        for (std::size_t k = 0; k < kids.Size(); ++k)
            out.children.push_back(kids[k].Int());
    }
    // Parent links, derived rather than declared — glTF only stores children.
    for (std::size_t i = 0; i < doc.nodes.size(); ++i)
        for (int c : doc.nodes[i].children)
            if (c >= 0 && std::size_t(c) < doc.nodes.size())
                doc.nodes[std::size_t(c)].parent = int(i);
    for (std::size_t i = 0; i < doc.nodes.size(); ++i)
        if (doc.nodes[i].parent < 0) doc.roots.push_back(int(i));

    return doc;
}

Document LoadGltfFile(const std::string& path, std::string& error) {
    Document doc;
    std::ifstream in(path, std::ios::binary);
    if (!in) { error = "gltf: cannot open " + path; return doc; }
    std::ostringstream ss;
    ss << in.rdbuf();
    const std::string text = ss.str();

    // Resolve a sibling .bin if the document names an external buffer.
    std::vector<std::uint8_t> bin;
    std::string probe_error;
    const json::Value root = json::Parse(text, probe_error);
    if (probe_error.empty()) {
        const json::Value& buffers = root["buffers"];
        for (std::size_t i = 0; i < buffers.Size(); ++i) {
            const std::string& uri = buffers[i]["uri"].Str();
            if (uri.empty() || uri.rfind("data:", 0) == 0) continue;
            const std::size_t slash = path.find_last_of('/');
            const std::string dir = slash == std::string::npos ? "" : path.substr(0, slash + 1);
            std::ifstream bf(dir + uri, std::ios::binary);
            if (!bf) { error = "gltf: cannot open buffer " + uri; return doc; }
            bin.assign(std::istreambuf_iterator<char>(bf), std::istreambuf_iterator<char>());
            break;
        }
    }
    Document doc2 = ParseGltf(text, bin, error);
    if (!error.empty()) return doc2;

    // Images that live in their own file next to the .gltf. ParseGltf recorded
    // their uris and left the slots empty because it has no filesystem access —
    // keeping that boundary is what makes the parser testable from a string.
    const std::size_t slash = path.find_last_of('/');
    const std::string dir = slash == std::string::npos ? "" : path.substr(0, slash + 1);
    for (std::size_t i = 0; i < doc2.image_uris.size(); ++i) {
        if (doc2.image_uris[i].empty()) continue;
        std::string ignored;
        doc2.images[i] = png::DecodeFile(dir + doc2.image_uris[i], ignored);
    }
    return doc2;
}

}  // namespace eng::gltf
