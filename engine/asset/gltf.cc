#include "engine/asset/gltf.h"

#include "engine/asset/jpeg.h"

#include "engine/asset/png.h"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <span>
#include <unordered_map>
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
        v.count = acc["count"].Int();
        v.component_type = acc["componentType"].Int();
        v.components = ComponentCount(acc["type"].Str());
        const int csize = ComponentSize(v.component_type);
        if (v.components == 0 || csize == 0) { Fail("unknown accessor type"); return v; }

        // SPARSE. A handful of elements overriding an otherwise shared buffer —
        // how an exporter ships a mesh that is mostly one thing with a few
        // vertices moved, without duplicating the whole array.
        //
        // Materialised into a densified copy rather than handled at read time.
        // A sparse view would have to binary-search its override list on every
        // element access, in the inner loop of every attribute; doing it once
        // costs a buffer and makes the rest of this file not care.
        if (acc.Has("sparse")) return Sparse(acc, v, csize);

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

    // Builds the densified copy a sparse accessor describes: the base values
    // where there are any, then the overrides written over the top.
    AccessorView Sparse(const json::Value& acc, AccessorView v, int csize) {
        const json::Value& sp = acc["sparse"];
        const int n = sp["count"].Int(0);
        const std::size_t elem = std::size_t(csize) * std::size_t(v.components);
        const std::size_t bytes = elem * std::size_t(v.count);
        if (v.count <= 0 || n < 0) { Fail("sparse accessor with no elements"); return v; }

        std::vector<std::uint8_t> dense(bytes, 0);

        // The base is OPTIONAL. Without one every element starts at zero, which
        // is the spec's answer and is how a morph target ships only the
        // vertices that actually move.
        const int bv_index = acc["bufferView"].Int(-1);
        if (bv_index >= 0) {
            const json::Value& bv = root_["bufferViews"][std::size_t(bv_index)];
            const int buf = bv["buffer"].Int(-1);
            if (buf < 0 || std::size_t(buf) >= buffers_.size()) {
                Fail("sparse accessor has a bad base buffer");
                return v;
            }
            const std::vector<std::uint8_t>& data = buffers_[std::size_t(buf)];
            std::size_t stride = std::size_t(bv["byteStride"].Int(0));
            if (stride == 0) stride = elem;
            const std::size_t off = std::size_t(bv["byteOffset"].Int(0)) +
                                    std::size_t(acc["byteOffset"].Int(0));
            for (int i = 0; i < v.count; ++i) {
                const std::size_t at = off + stride * std::size_t(i);
                if (at + elem > data.size()) {
                    Fail("sparse accessor's base runs past the buffer");
                    return v;
                }
                std::memcpy(&dense[elem * std::size_t(i)], data.data() + at, elem);
            }
        }

        if (n > 0) {
            const AccessorView idx = ViewOfBufferView(sp["indices"], 1, n);
            const AccessorView val = ViewOfBufferView(sp["values"], v.components, n);
            if (!idx.valid || !val.valid) {
                Fail("sparse accessor's index or value list runs past its buffer");
                return v;
            }
            for (int i = 0; i < n; ++i) {
                const std::uint32_t target = ReadIndex(idx, i);
                // An override aimed past the end is a corrupt file, not a
                // reason to write outside the array.
                if (target >= std::uint32_t(v.count)) {
                    Fail("sparse accessor overrides an element that is not there");
                    return v;
                }
                const std::uint8_t* src = val.base + val.stride * std::size_t(i);
                std::memcpy(&dense[elem * std::size_t(target)], src, elem);
            }
        }

        scratch_.push_back(std::move(dense));
        v.base = scratch_.back().data();
        v.stride = elem;  // densified, so tightly packed by construction
        v.valid = true;
        return v;
    }

    // A bufferView referenced directly, the way a sparse accessor's index and
    // value arrays are — they carry their own componentType and no count.
    // A view over a bufferView for a sparse accessor's index or value list.
    //
    // `count` IS NOT OPTIONAL, and leaving it out was a genuine out-of-bounds
    // read. This used to check only that the start offset was inside the
    // buffer; the caller then read `count` elements from it, with `count`
    // taken straight out of the file. A .glb claiming 9 overrides over a
    // bufferView holding 2 reads 7 elements past the end of the buffer, and
    // the offsets are entirely attacker-controlled.
    //
    // The main accessor path in View() has always had this check. Only this
    // one, reached solely by a sparse accessor, did not -- which is why it
    // survived: sparse is the rare path, and the base of a sparse accessor is
    // bounds-checked a few lines up, so the code around it looked careful.
    AccessorView ViewOfBufferView(const json::Value& spec, int components,
                                  int count) {
        AccessorView v;
        const int bv_index = spec["bufferView"].Int(-1);
        if (bv_index < 0) return v;
        const json::Value& bv = root_["bufferViews"][std::size_t(bv_index)];
        const int buf = bv["buffer"].Int(-1);
        if (buf < 0 || std::size_t(buf) >= buffers_.size()) return v;
        v.component_type = spec["componentType"].Int(kFloat);
        v.components = components;
        const int csize = ComponentSize(v.component_type);
        if (csize == 0) return v;
        v.stride = std::size_t(csize * components);
        const std::size_t off = std::size_t(bv["byteOffset"].Int(0)) +
                                std::size_t(spec["byteOffset"].Int(0));
        const std::vector<std::uint8_t>& data = buffers_[std::size_t(buf)];
        if (count < 0) return v;
        // The same bound View() uses: the last element has to end inside the
        // buffer, not merely start inside it.
        const std::size_t need =
            off + v.stride * std::size_t(count > 0 ? count - 1 : 0) + v.stride;
        if (count > 0 && (off >= data.size() || need > data.size())) return v;
        v.count = count;
        v.base = data.data() + off;
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
    // Densified sparse accessors. A vector of vectors is safe to grow while
    // views point into it: reallocating the outer one MOVES the inner vectors,
    // which transfers their heap buffers rather than copying them, so data()
    // stays put.
    std::vector<std::vector<std::uint8_t>> scratch_;
};

// A node's rest transform as SEPARATE components, which is what a skeleton
// needs: a clip interpolates rotations as quaternions, and a matrix cannot be
// decomposed back into one without guessing about shear.
//
// A node given as a bare "matrix" has no components to recover. glTF forbids
// that form for any node targeted by an animation, and a joint in a rest pose
// is exactly such a node, so falling back to identity here is the spec's own
// position rather than a shortcut.
anim::Transform NodeRest(const json::Value& n) {
    anim::Transform t;
    if (n.Has("translation")) {
        const json::Value& v = n["translation"];
        t.translation = Vec3{float(v[0].Number()), float(v[1].Number()),
                             float(v[2].Number())};
    }
    if (n.Has("rotation")) {
        const json::Value& v = n["rotation"];
        t.rotation = Normalize(Quat{float(v[0].Number()), float(v[1].Number()),
                                    float(v[2].Number()), float(v[3].Number(1.0))});
    }
    if (n.Has("scale")) {
        const json::Value& v = n["scale"];
        t.scale = Vec3{float(v[0].Number(1.0)), float(v[1].Number(1.0)),
                       float(v[2].Number(1.0))};
    }
    return t;
}

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
                // Sniffed from the BYTES, not from the declared mimeType. glTF
                // makes mimeType optional for a bufferView image and exporters
                // get it wrong when it is present; the magic number never lies.
                decoded = jpeg::IsJpeg(view) ? jpeg::Decode(view, ignored)
                                             : png::Decode(view, ignored);
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
        // Target names live on the MESH, not the primitive, and in "extras" --
        // glTF never standardised them, but every DCC tool writes them there
        // and a face rig with 52 unnamed targets is unusable.
        std::vector<std::string> target_names;
        {
            const json::Value& tn = meshes[mi]["extras"]["targetNames"];
            for (std::size_t i = 0; i < tn.Size(); ++i)
                target_names.push_back(tn[i].Str());
        }
        target_names.resize(64);  // so an unnamed target reads as "" not a crash
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
            AccessorView nrm, uv, tan;
            const bool has_normal = attrs.Has("NORMAL");
            const bool has_uv = attrs.Has("TEXCOORD_0");
            // TANGENT is vec4: xyz plus the bitangent's handedness in w, which
            // is the same layout the engine's VertexIn uses. Taken from the
            // file when it is there rather than regenerated, because an
            // exporter's tangents are the ones the normal map was BAKED
            // against -- regenerating them from a different triangulation or a
            // different smoothing choice produces a frame the map was not made
            // for, and the error shows up as lighting that swims across a seam.
            const bool has_tangent = attrs.Has("TANGENT");
            if (has_normal) { nrm = reader.View(attrs["NORMAL"].Int()); if (!nrm.valid) return doc; }
            if (has_uv) { uv = reader.View(attrs["TEXCOORD_0"].Int()); if (!uv.valid) return doc; }
            if (has_tangent) { tan = reader.View(attrs["TANGENT"].Int()); if (!tan.valid) return doc; }
            AccessorView joints, weights;
            // Both or neither. One without the other is a file that says which
            // joints influence a vertex but not by how much, or the reverse.
            const bool has_skin = attrs.Has("JOINTS_0") && attrs.Has("WEIGHTS_0");
            if (has_skin) {
                joints = reader.View(attrs["JOINTS_0"].Int());
                weights = reader.View(attrs["WEIGHTS_0"].Int());
                if (!joints.valid || !weights.valid) return doc;
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
                if (has_tangent) {
                    float t[4] = {1, 0, 0, 1};
                    Reader::ReadFloats(tan, i, t);
                    v.tangent = Vec4{t[0], t[1], t[2], t[3] < 0.0f ? -1.0f : 1.0f};
                }
                v.color = Vec4{1, 1, 1, 1};
                out.mesh.vertices.push_back(v);

                if (!has_skin) continue;
                // JOINTS_0 is an integer type (ubyte or ushort) that ReadFloats
                // has already widened; the cast back is exact for every index
                // a 16-bit palette can hold.
                float j[4] = {0, 0, 0, 0}, w[4] = {0, 0, 0, 0};
                Reader::ReadFloats(joints, i, j);
                Reader::ReadFloats(weights, i, w);
                anim::SkinVertex sv;
                for (int c = 0; c < anim::kMaxInfluences; ++c) {
                    sv.joints[c] = std::uint16_t(j[c]);
                    sv.weights[c] = w[c];
                }
                // Exporters round, and weights that do not sum to one shrink
                // the vertex toward the origin.
                anim::NormalizeWeights(&sv);
                out.skin.push_back(sv);
            }

            if (p.Has("indices")) {
                const AccessorView idx = reader.View(p["indices"].Int());
                if (!idx.valid) return doc;
                out.mesh.indices.reserve(std::size_t(idx.count));
                for (int i = 0; i < idx.count; ++i)
                    out.mesh.indices.push_back(Reader::ReadIndex(idx, i));
            } else {
                // Non-indexed: synthesise a trivial index buffer so the rest of
                // the engine only ever deals with indexed draws.
                out.mesh.indices.resize(std::size_t(pos.count));
                for (int i = 0; i < pos.count; ++i)
                    out.mesh.indices[std::size_t(i)] = std::uint32_t(i);
            }

            // --- morph targets ---------------------------------------------
            //
            // Each entry of "targets" is an attribute dictionary like the
            // primitive's own, but its accessors hold DELTAS. A target may
            // carry POSITION, NORMAL, both, or (legally) neither.
            const json::Value& targets = p["targets"];
            for (std::size_t ti = 0; ti < targets.Size(); ++ti) {
                const json::Value& t = targets[ti];
                anim::MorphTarget mt;
                if (t.Has("POSITION")) {
                    const AccessorView v = reader.View(t["POSITION"].Int());
                    if (!v.valid) return doc;
                    // A target shorter than the mesh would silently morph only
                    // the first part of it, which reads as the model tearing.
                    if (v.count != pos.count) {
                        error = "gltf: morph target has " +
                                std::to_string(v.count) +
                                " positions for a primitive of " +
                                std::to_string(pos.count);
                        return doc;
                    }
                    mt.positions.resize(std::size_t(v.count));
                    for (int i = 0; i < v.count; ++i) {
                        float f[4] = {0, 0, 0, 0};
                        Reader::ReadFloats(v, i, f);
                        mt.positions[std::size_t(i)] = Vec3{f[0], f[1], f[2]};
                    }
                }
                if (t.Has("NORMAL")) {
                    const AccessorView v = reader.View(t["NORMAL"].Int());
                    if (!v.valid) return doc;
                    if (v.count != pos.count) {
                        error = "gltf: morph target has the wrong normal count";
                        return doc;
                    }
                    mt.normals.resize(std::size_t(v.count));
                    for (int i = 0; i < v.count; ++i) {
                        float f[4] = {0, 0, 0, 0};
                        Reader::ReadFloats(v, i, f);
                        mt.normals[std::size_t(i)] = Vec3{f[0], f[1], f[2]};
                    }
                }
                mt.name = target_names[ti];
                out.morph_targets.push_back(std::move(mt));
            }
            // Defaults. glTF says an absent weights array means all zero, and
            // sizing it here means nothing downstream has to special-case the
            // empty case against the target count.
            out.morph_weights.assign(out.morph_targets.size(), 0.0f);
            const json::Value& mw = meshes[mi]["weights"];
            for (std::size_t i = 0;
                 i < mw.Size() && i < out.morph_weights.size(); ++i)
                out.morph_weights[i] = float(mw[i].Number());

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
        out.skin = n["skin"].Int(-1);
        const json::Value& nw = n["weights"];
        for (std::size_t k = 0; k < nw.Size(); ++k)
            out.morph_weights.push_back(float(nw[k].Number()));
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

    // --- skins -----------------------------------------------------------------
    // Built AFTER the nodes, because a joint's rest transform and its parent
    // both come from the node hierarchy.
    {
        const json::Value& skins = root["skins"];
        for (std::size_t i = 0; i < skins.Size(); ++i) {
            const json::Value& sk = skins[i];
            SkinDef def;
            def.name = sk["name"].Str();

            const json::Value& joints = sk["joints"];
            def.joint_nodes.reserve(joints.Size());
            for (std::size_t j = 0; j < joints.Size(); ++j)
                def.joint_nodes.push_back(joints[j].Int(-1));

            // Node index -> joint index, so a joint's parent link can be
            // expressed in the skeleton's own numbering. A joint whose node
            // parent is outside the skin becomes a root, which is correct: the
            // skin is only the part of the hierarchy it lists.
            std::unordered_map<int, int> joint_of;
            for (std::size_t j = 0; j < def.joint_nodes.size(); ++j)
                joint_of[def.joint_nodes[j]] = int(j);

            AccessorView ibm;
            const int ibm_index = sk["inverseBindMatrices"].Int(-1);
            if (ibm_index >= 0) {
                ibm = reader.View(ibm_index);
                if (!ibm.valid) return doc;
            }

            def.skeleton.joints.resize(def.joint_nodes.size());
            for (std::size_t j = 0; j < def.joint_nodes.size(); ++j) {
                const int node = def.joint_nodes[j];
                anim::Joint& out = def.skeleton.joints[j];
                if (node < 0 || std::size_t(node) >= doc.nodes.size()) {
                    error = "gltf: skin joint refers to a node that is not there";
                    return doc;
                }
                out.name = doc.nodes[std::size_t(node)].name;
                const auto it = joint_of.find(doc.nodes[std::size_t(node)].parent);
                out.parent = it == joint_of.end() ? -1 : it->second;
                out.rest = NodeRest(nodes[std::size_t(node)]);

                // Absent inverseBindMatrices means identity, per the spec: the
                // mesh is already in each joint's space.
                if (ibm_index >= 0 && int(j) < ibm.count) {
                    float m[16];
                    Reader::ReadFloats(ibm, int(j), m);
                    // glTF matrices are column-major, and so is Mat4, so this
                    // is a straight copy rather than a transpose.
                    out.inverse_bind = Mat4{{{m[0], m[1], m[2], m[3]},
                                             {m[4], m[5], m[6], m[7]},
                                             {m[8], m[9], m[10], m[11]},
                                             {m[12], m[13], m[14], m[15]}}};
                }
            }
            if (!def.skeleton.Finalize()) {
                error = "gltf: skin " + std::to_string(i) + " has a cyclic hierarchy";
                return doc;
            }
            doc.skins.push_back(std::move(def));
        }
    }

    // --- animations --------------------------------------------------------------
    {
        const json::Value& anims = root["animations"];
        for (std::size_t i = 0; i < anims.Size(); ++i) {
            const json::Value& a = anims[i];
            AnimationDef def;
            def.name = a["name"].Str();
            const json::Value& samplers = a["samplers"];
            const json::Value& channels = a["channels"];

            for (std::size_t c = 0; c < channels.Size(); ++c) {
                const json::Value& ch = channels[c];
                const json::Value& target = ch["target"];
                const std::string& path = target["path"].Str();

                NodeChannel out;
                out.node = target["node"].Int(-1);
                if (path == "translation") out.path = anim::Path::Translation;
                else if (path == "rotation") out.path = anim::Path::Rotation;
                else if (path == "scale") out.path = anim::Path::Scale;
                else if (path == "weights") out.path = anim::Path::Weights;
                else continue;  // an extension path this reader does not know

                const int si = ch["sampler"].Int(-1);
                if (si < 0 || std::size_t(si) >= samplers.Size()) continue;
                const json::Value& sampler = samplers[std::size_t(si)];
                const std::string& interp = sampler["interpolation"].Str();
                if (interp == "STEP") out.interp = anim::Interp::Step;
                else if (interp == "CUBICSPLINE") out.interp = anim::Interp::CubicSpline;
                else out.interp = anim::Interp::Linear;  // the spec's default

                const AccessorView in = reader.View(sampler["input"].Int(-1));
                const AccessorView vals = reader.View(sampler["output"].Int(-1));
                if (!in.valid || !vals.valid) return doc;

                out.times.reserve(std::size_t(in.count));
                for (int k = 0; k < in.count; ++k) {
                    float t[4] = {0, 0, 0, 0};
                    Reader::ReadFloats(in, k, t);
                    out.times.push_back(t[0]);
                    def.duration = std::max(def.duration, t[0]);
                }
                out.values.reserve(std::size_t(vals.count) * 4);
                for (int k = 0; k < vals.count; ++k) {
                    float v[4] = {0, 0, 0, 0};
                    Reader::ReadFloats(vals, k, v);
                    for (int q = 0; q < vals.components; ++q) out.values.push_back(v[q]);
                }
                def.channels.push_back(std::move(out));
            }
            doc.animations.push_back(std::move(def));
        }
    }

    return doc;
}

anim::MorphTrack Document::MakeMorphTrack(int animation, int node) const {
    anim::MorphTrack track;
    if (animation < 0 || std::size_t(animation) >= animations.size()) return track;
    if (node < 0 || std::size_t(node) >= nodes.size()) return track;

    // How many targets the node's mesh has. A weights channel stores a flat run
    // of floats with no count of its own -- glTF puts the count on the mesh, so
    // reading the channel without the node it drives cannot tell where one key
    // ends and the next begins.
    int targets = 0;
    for (int pi : nodes[std::size_t(node)].primitives)
        if (pi >= 0 && std::size_t(pi) < primitives.size())
            targets = std::max(
                targets, int(primitives[std::size_t(pi)].morph_targets.size()));
    if (targets == 0) return track;

    for (const NodeChannel& ch : animations[std::size_t(animation)].channels) {
        if (ch.node != node || ch.path != anim::Path::Weights) continue;
        track.interp = ch.interp;
        track.targets = targets;
        track.times = ch.times;
        track.values = ch.values;
        track.duration = animations[std::size_t(animation)].duration;
        break;  // one weights channel per node; a second would be a conflict
    }
    return track;
}

anim::Clip Document::MakeClip(int animation, int skin) const {
    anim::Clip clip;
    if (animation < 0 || std::size_t(animation) >= animations.size()) return clip;
    if (skin < 0 || std::size_t(skin) >= skins.size()) return clip;

    const AnimationDef& src = animations[std::size_t(animation)];
    const SkinDef& sk = skins[std::size_t(skin)];
    clip.name = src.name;
    clip.duration = src.duration;

    std::unordered_map<int, int> joint_of;
    for (std::size_t j = 0; j < sk.joint_nodes.size(); ++j)
        joint_of[sk.joint_nodes[j]] = int(j);

    for (const NodeChannel& nc : src.channels) {
        const auto it = joint_of.find(nc.node);
        // A channel aimed at a node this skin does not use is dropped rather
        // than mapped to joint 0 — one file can hold two characters, and
        // silently applying one's walk cycle to the other is worse than
        // applying nothing.
        if (it == joint_of.end()) continue;
        // Morph weights ride on the same channel list but are not a joint
        // property; MakeMorphTrack is where they belong.
        if (nc.path == anim::Path::Weights) continue;
        anim::Channel out;
        out.joint = it->second;
        out.path = nc.path;
        out.interp = nc.interp;
        out.times = nc.times;
        out.values = nc.values;
        clip.channels.push_back(std::move(out));
    }
    return clip;
}

bool IsGlb(std::span<const std::uint8_t> bytes) {
    return bytes.size() >= 12 && bytes[0] == 'g' && bytes[1] == 'l' &&
           bytes[2] == 'T' && bytes[3] == 'F';
}

Document ParseGlb(std::span<const std::uint8_t> bytes, std::string& error) {
    Document doc;
    if (!IsGlb(bytes)) {
        error = "glb: bad magic";
        return doc;
    }
    auto read32 = [&](std::size_t at) {
        return std::uint32_t(bytes[at]) | (std::uint32_t(bytes[at + 1]) << 8) |
               (std::uint32_t(bytes[at + 2]) << 16) |
               (std::uint32_t(bytes[at + 3]) << 24);
    };
    const std::uint32_t version = read32(4);
    if (version != 2) {
        error = "glb: version " + std::to_string(version) + ", expected 2";
        return doc;
    }
    // The header's total length is advisory — a file may be longer. Trusting it
    // over the actual size is how a truncated download reads past its end.
    const std::size_t total = std::min(std::size_t(read32(8)), bytes.size());

    std::string json;
    std::vector<std::uint8_t> bin;
    std::size_t at = 12;
    while (at + 8 <= total) {
        const std::uint32_t len = read32(at);
        const std::uint32_t type = read32(at + 4);
        const std::size_t body = at + 8;
        if (body + len > total) {
            error = "glb: chunk runs past the end of the file";
            return doc;
        }
        if (type == 0x4E4F534Au) {  // 'JSON'
            json.assign(reinterpret_cast<const char*>(bytes.data() + body), len);
        } else if (type == 0x004E4942u) {  // 'BIN\0'
            bin.assign(bytes.begin() + std::ptrdiff_t(body),
                       bytes.begin() + std::ptrdiff_t(body + len));
        }
        // Anything else is an extension chunk and is skipped, per the spec.
        // Chunks are four-byte aligned.
        at = body + len;
        at += (4 - (at % 4)) % 4;
    }
    if (json.empty()) {
        error = "glb: no JSON chunk";
        return doc;
    }
    return ParseGltf(json, bin, error);
}

namespace {

// One file, either format, chosen by its first bytes. The extension is not
// consulted: a .png that is really a jpeg is common enough in asset pipelines
// that trusting the name means a black texture and no error.
Texture2D DecodeImageFile(const std::string& path, std::string& error) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        error = "image: cannot open " + path;
        return {};
    }
    // Braces, not parens: with parens this is the most vexing parse and
    // declares a function taking two iterators.
    const std::vector<std::uint8_t> bytes{std::istreambuf_iterator<char>(f),
                                          std::istreambuf_iterator<char>()};
    return jpeg::IsJpeg(bytes) ? jpeg::Decode(bytes, error)
                               : png::Decode(bytes, error);
}

}  // namespace

Document LoadGltfFile(const std::string& path, std::string& error) {
    Document doc;
    std::ifstream in(path, std::ios::binary);
    if (!in) { error = "gltf: cannot open " + path; return doc; }
    std::ostringstream ss;
    ss << in.rdbuf();
    const std::string text = ss.str();

    // Sniffed by content, not by extension: a .gltf that is really a glb is a
    // real thing that happens, and the magic is four bytes.
    {
        const auto* raw = reinterpret_cast<const std::uint8_t*>(text.data());
        if (IsGlb(std::span<const std::uint8_t>(raw, text.size())))
            return ParseGlb(std::span<const std::uint8_t>(raw, text.size()), error);
    }

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
        doc2.images[i] = DecodeImageFile(dir + doc2.image_uris[i], ignored);
    }
    return doc2;
}

}  // namespace eng::gltf
