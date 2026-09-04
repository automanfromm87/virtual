#include "engine/asset/json.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>

namespace eng::json {
namespace {

const Value& NullValue() {
    static const Value kNull;
    return kNull;
}

}  // namespace

const Value& Value::operator[](std::size_t i) const {
    return i < array_.size() ? array_[i] : NullValue();
}

const Value& Value::operator[](std::string_view key) const {
    const auto it = object_.find(key);
    return it == object_.end() ? NullValue() : it->second;
}

bool Value::Has(std::string_view key) const {
    return object_.find(key) != object_.end();
}

class Parser {
  public:
    Parser(std::string_view text, std::string& error) : s_(text), error_(error) {}

    // Counts nesting for the check in ParseValue, and un-counts it however the
    // parse leaves -- there are early returns on every failure path.
    struct DepthGuard {
        int& d;
        explicit DepthGuard(int& depth) : d(depth) { ++d; }
        ~DepthGuard() { --d; }
    };

    // A GUARD, not a style choice. ParseValue calls ParseArray and ParseObject
    // and both call back into ParseValue, so nesting in the INPUT becomes
    // nesting on the C stack one frame per bracket. A few kilobytes of "[[[[["
    // overflows it and the process dies before any error can be returned --
    // and JSON is the one format here that is routinely handed untrusted bytes.
    //
    // 128 is far past anything a real document reaches; glTF's deepest
    // structure is about six.
    static constexpr int kMaxDepth = 128;

    Value ParseValue() {
        SkipSpace();
        if (Done()) return Fail("unexpected end of input");
        if (depth_ >= kMaxDepth) return Fail("nested too deeply");
        const DepthGuard guard(depth_);
        switch (Peek()) {
            case '{': return ParseObject();
            case '[': return ParseArray();
            case '"': return ParseString();
            case 't': return ParseLiteral("true", true);
            case 'f': return ParseLiteral("false", false);
            case 'n': return ParseNull();
            default: return ParseNumber();
        }
    }

    void SkipSpace() {
        while (!Done() && (Peek() == ' ' || Peek() == '\t' || Peek() == '\n' ||
                           Peek() == '\r'))
            ++i_;
    }
    [[nodiscard]] bool Done() const { return i_ >= s_.size(); }
    [[nodiscard]] char Peek() const { return s_[i_]; }
    [[nodiscard]] bool Failed() const { return !error_.empty(); }

  private:
    Value Fail(const char* what) {
        if (error_.empty())
            error_ = std::string("json: ") + what + " at offset " + std::to_string(i_);
        return Value{};
    }

    Value ParseNull() {
        if (s_.compare(i_, 4, "null") != 0) return Fail("expected null");
        i_ += 4;
        return Value{};
    }

    Value ParseLiteral(const char* text, bool value) {
        const std::size_t n = std::string_view(text).size();
        if (s_.compare(i_, n, text) != 0) return Fail("expected a literal");
        i_ += n;
        Value v;
        v.type_ = Value::Type::Bool;
        v.bool_ = value;
        return v;
    }

    Value ParseNumber() {
        const std::size_t start = i_;
        if (!Done() && (Peek() == '-' || Peek() == '+')) ++i_;
        while (!Done() && ((Peek() >= '0' && Peek() <= '9') || Peek() == '.' ||
                           Peek() == 'e' || Peek() == 'E' || Peek() == '-' ||
                           Peek() == '+'))
            ++i_;
        if (i_ == start) return Fail("expected a number");
        Value v;
        v.type_ = Value::Type::Number;
        // strtod over the exact span. std::stod would need a null-terminated
        // copy of every number in the file.
        const std::string text(s_.substr(start, i_ - start));
        v.number_ = std::strtod(text.c_str(), nullptr);
        return v;
    }

    Value ParseString() {
        if (Peek() != '"') return Fail("expected a string");
        ++i_;
        Value v;
        v.type_ = Value::Type::String;
        while (true) {
            if (Done()) return Fail("unterminated string");
            const char c = s_[i_++];
            if (c == '"') break;
            if (c != '\\') {
                v.string_ += c;
                continue;
            }
            if (Done()) return Fail("unterminated escape");
            const char e = s_[i_++];
            switch (e) {
                case '"': v.string_ += '"'; break;
                case '\\': v.string_ += '\\'; break;
                case '/': v.string_ += '/'; break;
                case 'b': v.string_ += '\b'; break;
                case 'f': v.string_ += '\f'; break;
                case 'n': v.string_ += '\n'; break;
                case 'r': v.string_ += '\r'; break;
                case 't': v.string_ += '\t'; break;
                case 'u': {
                    // \uXXXX. Encoded as UTF-8; surrogate pairs are left as the
                    // replacement character rather than silently mangled.
                    if (i_ + 4 > s_.size()) return Fail("truncated \\u escape");
                    unsigned code = 0;
                    for (int k = 0; k < 4; ++k) {
                        const char h = s_[i_++];
                        code <<= 4;
                        if (h >= '0' && h <= '9') code |= unsigned(h - '0');
                        else if (h >= 'a' && h <= 'f') code |= unsigned(h - 'a' + 10);
                        else if (h >= 'A' && h <= 'F') code |= unsigned(h - 'A' + 10);
                        else return Fail("bad hex in \\u escape");
                    }
                    if (code >= 0xD800 && code <= 0xDFFF) code = 0xFFFD;
                    if (code < 0x80) {
                        v.string_ += char(code);
                    } else if (code < 0x800) {
                        v.string_ += char(0xC0 | (code >> 6));
                        v.string_ += char(0x80 | (code & 0x3F));
                    } else {
                        v.string_ += char(0xE0 | (code >> 12));
                        v.string_ += char(0x80 | ((code >> 6) & 0x3F));
                        v.string_ += char(0x80 | (code & 0x3F));
                    }
                    break;
                }
                default: return Fail("unknown escape");
            }
        }
        return v;
    }

    Value ParseArray() {
        ++i_;  // '['
        Value v;
        v.type_ = Value::Type::Array;
        SkipSpace();
        if (!Done() && Peek() == ']') { ++i_; return v; }
        while (true) {
            v.array_.push_back(ParseValue());
            if (Failed()) return Value{};
            SkipSpace();
            if (Done()) return Fail("unterminated array");
            if (Peek() == ',') { ++i_; continue; }
            if (Peek() == ']') { ++i_; return v; }
            return Fail("expected ',' or ']'");
        }
    }

    Value ParseObject() {
        ++i_;  // '{'
        Value v;
        v.type_ = Value::Type::Object;
        SkipSpace();
        if (!Done() && Peek() == '}') { ++i_; return v; }
        while (true) {
            SkipSpace();
            const Value key = ParseString();
            if (Failed()) return Value{};
            SkipSpace();
            if (Done() || Peek() != ':') return Fail("expected ':'");
            ++i_;
            Value val = ParseValue();
            if (Failed()) return Value{};
            v.object_.emplace(key.string_, std::move(val));
            SkipSpace();
            if (Done()) return Fail("unterminated object");
            if (Peek() == ',') { ++i_; continue; }
            if (Peek() == '}') { ++i_; return v; }
            return Fail("expected ',' or '}'");
        }
    }

    std::string_view s_;
    std::string& error_;
    int depth_ = 0;
    std::size_t i_ = 0;
};

Value Parse(std::string_view text, std::string& error) {
    error.clear();
    Parser p(text, error);
    Value v = p.ParseValue();
    if (!error.empty()) return Value{};
    p.SkipSpace();
    // Trailing content means the document was not what the caller thought it
    // was — silently ignoring it hides truncated or concatenated files.
    if (!p.Done()) {
        error = "json: trailing content after the top-level value";
        return Value{};
    }
    return v;
}

bool DecodeBase64(std::string_view in, std::vector<std::uint8_t>& out) {
    auto sextet = [](char c) -> int {
        if (c >= 'A' && c <= 'Z') return c - 'A';
        if (c >= 'a' && c <= 'z') return c - 'a' + 26;
        if (c >= '0' && c <= '9') return c - '0' + 52;
        if (c == '+') return 62;
        if (c == '/') return 63;
        return -1;
    };
    out.clear();
    out.reserve(in.size() * 3 / 4);
    int acc = 0, bits = 0;
    for (char c : in) {
        if (c == '=' ) break;
        if (c == '\n' || c == '\r' || c == ' ') continue;
        const int s = sextet(c);
        if (s < 0) return false;
        acc = (acc << 6) | s;
        bits += 6;
        if (bits >= 8) {
            bits -= 8;
            out.push_back(std::uint8_t((acc >> bits) & 0xFF));
        }
    }
    return true;
}

// ----------------------------------------------------------------- Writer --

void Writer::Separate() {
    if (after_key_) {
        after_key_ = false;
        return;
    }
    if (nonempty_.empty()) return;
    if (nonempty_.back()) out_ += ',';
    nonempty_.back() = true;
    if (!inline_.back()) {
        out_ += '\n';
        Indent();
    } else if (out_.back() == ',') {
        out_ += ' ';
    }
}

void Writer::Indent() { out_.append(nonempty_.size() * 2, ' '); }

void Writer::BeginObject() {
    Separate();
    out_ += '{';
    nonempty_.push_back(false);
    inline_.push_back(false);
}

void Writer::EndObject() {
    const bool had = nonempty_.back();
    nonempty_.pop_back();
    inline_.pop_back();
    if (had) {
        out_ += '\n';
        Indent();
    }
    out_ += '}';
}

void Writer::BeginArray() {
    Separate();
    out_ += '[';
    nonempty_.push_back(false);
    inline_.push_back(false);
}

void Writer::BeginInlineArray() {
    Separate();
    out_ += '[';
    nonempty_.push_back(false);
    inline_.push_back(true);
}

void Writer::EndArray() {
    const bool had = nonempty_.back();
    const bool inl = inline_.back();
    nonempty_.pop_back();
    inline_.pop_back();
    if (had && !inl) {
        out_ += '\n';
        Indent();
    }
    out_ += ']';
}

void Writer::Key(std::string_view k) {
    Separate();
    // The key's own string write must not separate again — the call above
    // already placed the comma and the newline for this whole pair.
    after_key_ = true;
    Value(k);
    out_ += ": ";
    after_key_ = true;  // and neither does the value that follows the colon
}

void Writer::Value(double v) {
    Separate();
    // JSON has no NaN and no infinity, and printf writes them as bare `nan` and
    // `inf` — a file that no parser will read back, produced without complaint.
    // A simulation that has gone non-finite is already broken; emitting null
    // keeps the file loadable and makes the damage visible in it.
    if (!std::isfinite(v)) {
        out_ += "null";
        return;
    }
    // %.9g round-trips a float exactly and still writes 0.5 as "0.5" rather
    // than "0.500000000". A saved scene is meant to be edited by hand.
    char buf[40];
    std::snprintf(buf, sizeof buf, "%.9g", v);
    out_ += buf;
}

void Writer::Value(int v) {
    Separate();
    out_ += std::to_string(v);
}

void Writer::Value(bool v) {
    Separate();
    out_ += v ? "true" : "false";
}

void Writer::Value(std::string_view s) {
    Separate();
    out_ += '"';
    for (char c : s) {
        switch (c) {
            case '"': out_ += "\\\""; break;
            case '\\': out_ += "\\\\"; break;
            case '\n': out_ += "\\n"; break;
            case '\t': out_ += "\\t"; break;
            case '\r': out_ += "\\r"; break;
            default:
                if (static_cast<unsigned char>(c) < 0x20) {
                    char esc[8];
                    std::snprintf(esc, sizeof esc, "\\u%04x", c);
                    out_ += esc;
                } else {
                    out_ += c;
                }
        }
    }
    out_ += '"';
}

void Writer::Null() {
    Separate();
    out_ += "null";
}

void Writer::Vec(const float* v, int n) {
    BeginInlineArray();
    for (int i = 0; i < n; ++i) Value(double(v[i]));
    EndArray();
}

std::string Writer::Take() {
    out_ += '\n';
    return std::move(out_);
}

}  // namespace eng::json
