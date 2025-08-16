#include "fix_bridge.h"
#include "../../cpp_fix_codec/fix_parser.h"
#include "../../cpp_fix_codec/fix_builder.h"
#include <cstring>
#include <memory>
#include <sstream>

// Parser wrapper
class FixParserWrapper {
public:
    GroupDefs defs;
    std::unique_ptr<FixMessage> msg;
    
    FixParserWrapper(int max_message_size) : msg(std::make_unique<FixMessage>(max_message_size)) {
        // Initialize group definitions for common message types
        // This would be expanded based on actual FIX specification needs
        defs.add("D", {GroupDef{453, 448}}); // NewOrderSingle with parties group
        defs.add("8", {GroupDef{453, 448}}); // ExecutionReport with parties group
    }
    
    bool parse(const char* data, int length) {
        std::istringstream stream(std::string(data, length));
        try {
            FixMessage::parse(stream, *msg, defs);
            return true;
        } catch (...) {
            return false;
        }
    }
};

extern "C" {

// Parser functions
FixParserHandle fix_parser_create(int max_message_size) {
    return new FixParserWrapper(max_message_size);
}

void fix_parser_destroy(FixParserHandle handle) {
    delete static_cast<FixParserWrapper*>(handle);
}

bool fix_parser_parse(FixParserHandle handle, const char* data, int length, FixMessageHandle* msg) {
    auto parser = static_cast<FixParserWrapper*>(handle);
    if (parser->parse(data, length)) {
        *msg = parser->msg.get();
        return true;
    }
    return false;
}

// Message functions
FixMessageHandle fix_message_create() {
    return new FixMessage();
}

void fix_message_destroy(FixMessageHandle handle) {
    // Don't delete if it's from parser (managed by parser)
    // This is a simplified approach - in production we'd track ownership
}

const char* fix_message_get_field(FixMessageHandle handle, uint32_t tag) {
    auto msg = static_cast<FixMessage*>(handle);
    try {
        auto view = msg->getString(tag);
        // This is unsafe as string_view doesn't guarantee null termination
        // In production, we'd need proper string management
        static thread_local std::string buffer;
        buffer = std::string(view);
        return buffer.c_str();
    } catch (...) {
        return nullptr;
    }
}

int fix_message_get_int_field(FixMessageHandle handle, uint32_t tag) {
    auto msg = static_cast<FixMessage*>(handle);
    try {
        return msg->getInt(tag);
    } catch (...) {
        return 0;
    }
}

double fix_message_get_double_field(FixMessageHandle handle, uint32_t tag) {
    auto msg = static_cast<FixMessage*>(handle);
    try {
        return msg->getDouble(tag);
    } catch (...) {
        return 0.0;
    }
}

bool fix_message_has_field(FixMessageHandle handle, uint32_t tag) {
    auto msg = static_cast<FixMessage*>(handle);
    return msg->hasTag(tag);
}

int fix_message_get_all_fields(FixMessageHandle handle, FixField* fields, int max_fields) {
    auto msg = static_cast<FixMessage*>(handle);
    int count = 0;
    
    for (auto tag : msg->tags()) {
        if (count >= max_fields) break;
        
        fields[count].tag = tag;
        auto value = msg->getString(tag);
        strncpy(fields[count].value, value.data(), std::min(value.size(), size_t(255)));
        fields[count].value[255] = '\0';
        count++;
    }
    
    return count;
}

// Builder functions
FixBuilderHandle fix_builder_create() {
    return new FixBuilder();
}

void fix_builder_destroy(FixBuilderHandle handle) {
    delete static_cast<FixBuilder*>(handle);
}

void fix_builder_reset(FixBuilderHandle handle) {
    static_cast<FixBuilder*>(handle)->reset();
}

void fix_builder_add_field(FixBuilderHandle handle, uint32_t tag, const char* value) {
    static_cast<FixBuilder*>(handle)->addField(tag, value);
}

void fix_builder_add_int_field(FixBuilderHandle handle, uint32_t tag, int value) {
    static_cast<FixBuilder*>(handle)->addField(tag, value);
}

void fix_builder_add_double_field(FixBuilderHandle handle, uint32_t tag, double value) {
    static_cast<FixBuilder*>(handle)->addField(tag, value);
}

int fix_builder_build(FixBuilderHandle handle, char* buffer, int buffer_size) {
    auto builder = static_cast<FixBuilder*>(handle);
    std::ostringstream stream;
    builder->writeTo(stream);
    
    std::string result = stream.str();
    if (result.size() >= buffer_size) {
        return -1; // Buffer too small
    }
    
    strcpy(buffer, result.c_str());
    return result.size();
}

// Simplified session implementation
class FixSessionWrapper {
public:
    FixSessionConfig config;
    FixMessageCallback callback;
    void* user_data;
    bool logged_on = false;
    
    FixSessionWrapper(const FixSessionConfig* cfg, FixMessageCallback cb, void* data) 
        : config(*cfg), callback(cb), user_data(data) {}
    
    bool send(const char* msg_type, FixBuilder* builder) {
        // In production, this would interface with the actual FIX engine
        // For now, just a placeholder
        return logged_on;
    }
};

FixSessionHandle fix_session_create(const FixSessionConfig* config, FixMessageCallback callback, void* user_data) {
    return new FixSessionWrapper(config, callback, user_data);
}

void fix_session_destroy(FixSessionHandle handle) {
    delete static_cast<FixSessionWrapper*>(handle);
}

bool fix_session_send(FixSessionHandle handle, const char* msg_type, FixBuilderHandle builder) {
    auto session = static_cast<FixSessionWrapper*>(handle);
    auto fix_builder = static_cast<FixBuilder*>(builder);
    return session->send(msg_type, fix_builder);
}

bool fix_session_is_logged_on(FixSessionHandle handle) {
    return static_cast<FixSessionWrapper*>(handle)->logged_on;
}

void fix_session_disconnect(FixSessionHandle handle) {
    static_cast<FixSessionWrapper*>(handle)->logged_on = false;
}

} // extern "C"