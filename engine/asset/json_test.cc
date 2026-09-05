// No test framework — from scratch means from scratch.
#include "engine/asset/json.h"

#include <cmath>
#include <cstdio>

namespace {

int g_failures = 0;

void Fail(const char* what, int line) {
    std::fprintf(stderr, "json_test.cc:%d  %s\n", line, what);
    ++g_failures;
}

#define CHECK(cond) \
    do { if (!(cond)) Fail(#cond, __LINE__); } while (0)

}  // namespace

int main() {
    using namespace eng::json;
    std::string err;

    {
        const Value v = Parse(R"({"a":1,"b":[2,3.5,-4e2],"c":"hi","d":true,"e":null})", err);
        CHECK(err.empty());
        CHECK(v.IsObject());
        CHECK(v["a"].Int() == 1);
        CHECK(v["b"].Size() == 3);
        CHECK(v["b"][1].Number() == 3.5);
        CHECK(v["b"][2].Number() == -400.0);
        CHECK(v["c"].Str() == "hi");
        CHECK(v["d"].Bool());
        CHECK(v["e"].IsNull());
    }

    {
        // Missing keys and out-of-range indices must be SAFE. A glTF document
        // is mostly optional fields; if every read could dangle, the reader
        // would be nothing but null checks.
        const Value v = Parse(R"({"a":1})", err);
        CHECK(err.empty());
        CHECK(v["nope"].IsNull());
        CHECK(v["nope"]["deeper"][7].IsNull());
        CHECK(v["nope"].Int(42) == 42);
        CHECK(!v.Has("nope"));
        CHECK(v.Has("a"));
    }

    {
        const Value v = Parse(R"("a\"b\\c\ndé")", err);
        CHECK(err.empty());
        CHECK(v.Str() == "a\"b\\c\nd\xc3\xa9");
    }

    {
        // Malformed input must REPORT, not guess. Silently accepting a
        // truncated file is how a corrupt asset becomes a rendering mystery.
        const char* bad[] = {"{", "[1,2", R"({"a" 1})", "tru", "{'a':1}", ""};
        for (const char* b : bad) {
            (void)Parse(b, err);
            CHECK(!err.empty());
        }
        // Trailing junk after a complete value is also an error.
        (void)Parse("{} extra", err);
        CHECK(!err.empty());
    }

    {
        std::vector<std::uint8_t> out;
        CHECK(DecodeBase64("TWFu", out));
        CHECK(out.size() == 3 && out[0] == 'M' && out[1] == 'a' && out[2] == 'n');
        CHECK(DecodeBase64("TWE=", out));
        CHECK(out.size() == 2 && out[0] == 'M' && out[1] == 'a');
        CHECK(DecodeBase64("TQ==", out));
        CHECK(out.size() == 1 && out[0] == 'M');
        CHECK(!DecodeBase64("!!!!", out));
    }

    // --- the writer never emits something no parser will read ------------------
    {
        Writer w;
        w.BeginObject();
        w.Key("nan");
        w.Value(std::nan(""));
        w.Key("inf");
        w.Value(1.0 / 0.0);
        w.Key("ok");
        w.Value(0.5);
        // A pointer would otherwise bind to the bool overload and write `true`.
        w.Key("str");
        w.Value("box");
        w.EndObject();
        const std::string text = w.Take();

        std::string error;
        const Value v = Parse(text, error);
        CHECK(error.empty());
        CHECK(v["nan"].IsNull());
        CHECK(v["inf"].IsNull());
        CHECK(std::fabs(v["ok"].Number() - 0.5) < 1e-12);
        CHECK(v["str"].IsString() && v["str"].Str() == "box");
    }

    // --- deep nesting is refused, not crashed on -------------------------------
    {
        std::printf("deep nesting\n");
        // ParseValue calls ParseArray which calls ParseValue: nesting in the
        // INPUT is nesting on the C stack, one frame per bracket. Without a
        // limit a few kilobytes of brackets takes the process down before any
        // error can be returned -- and this is the one format here that is
        // routinely handed bytes from somewhere else.
        const auto nested = [](int n) {
            return std::string(std::size_t(n), '[') + std::string(std::size_t(n), ']');
        };
        std::string error;
        const Value shallow = Parse(nested(64), error);
        std::printf("    64 deep: error \"%s\"\n", error.c_str());
        CHECK(error.empty());
        CHECK(shallow.IsArray());

        error.clear();
        Parse(nested(100000), error);
        std::printf("    100000 deep: error \"%s\"\n", error.c_str());
        CHECK(!error.empty());
        CHECK(error.find("deep") != std::string::npos);

        // AND THE COUNTER UNWINDS. A depth that is incremented on the way in
        // and not decremented on the way out makes a long FLAT document fail
        // as though it were deep -- which would be a worse bug than the one
        // being fixed, and it would only show on real files.
        error.clear();
        std::string flat = "[";
        for (int i = 0; i < 5000; ++i) flat += (i ? ",[1,[2]]" : "[1,[2]]");
        flat += "]";
        const Value wide = Parse(flat, error);
        std::printf("    5000 siblings each 3 deep: error \"%s\"\n", error.c_str());
        CHECK(error.empty());
        CHECK(wide.Size() == 5000);
    }

    // --- strict accessors fail loudly on required fields ----------------------
    {
        std::printf("strict accessors\n");
        std::string error;
        const Value doc =
            Parse(R"({"name":"fox","count":3,"tags":["a"]})", error);
        CHECK(error.empty());

        const Value* name = doc.Find("name");
        CHECK(name != nullptr && name->Str() == "fox");
        CHECK(doc.Find("missing") == nullptr);
        CHECK(doc.Find("name") != nullptr);  // present, any type
        const Value* count = doc.Find("count");
        CHECK(count != nullptr && count->Int(-1) == 3);

        const Value* out = nullptr;
        CHECK(doc.At("count", Value::Type::Number, out, error));
        CHECK(out == count);
        CHECK(!doc.At("absent", Value::Type::Number, out, error));
        std::printf("    missing field: \"%s\"\n", error.c_str());
        CHECK(error.find("absent") != std::string::npos);
        CHECK(!doc.At("name", Value::Type::Number, out, error));
        std::printf("    wrong type: \"%s\"\n", error.c_str());
        CHECK(error.find("name") != std::string::npos &&
              error.find("number") != std::string::npos);
        // The lenient path is untouched: optional reads still yield fallbacks.
        CHECK(doc["absent"].Number(-2.5) == -2.5);
    }

    if (g_failures == 0) std::printf("json_test: all checks passed\n");
    return g_failures == 0 ? 0 : 1;
}
