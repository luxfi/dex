#ifndef LX_FIX_BRIDGE_H
#define LX_FIX_BRIDGE_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stdbool.h>

// FIX message handle (opaque pointer)
typedef void* FixMessageHandle;
typedef void* FixParserHandle;
typedef void* FixBuilderHandle;

// FIX field structure
typedef struct {
    uint32_t tag;
    char value[256];
} FixField;

// Parser functions
FixParserHandle fix_parser_create(int max_message_size);
void fix_parser_destroy(FixParserHandle handle);
bool fix_parser_parse(FixParserHandle handle, const char* data, int length, FixMessageHandle* msg);

// Message functions
FixMessageHandle fix_message_create();
void fix_message_destroy(FixMessageHandle handle);
const char* fix_message_get_field(FixMessageHandle handle, uint32_t tag);
int fix_message_get_int_field(FixMessageHandle handle, uint32_t tag);
double fix_message_get_double_field(FixMessageHandle handle, uint32_t tag);
bool fix_message_has_field(FixMessageHandle handle, uint32_t tag);
int fix_message_get_all_fields(FixMessageHandle handle, FixField* fields, int max_fields);

// Builder functions
FixBuilderHandle fix_builder_create();
void fix_builder_destroy(FixBuilderHandle handle);
void fix_builder_reset(FixBuilderHandle handle);
void fix_builder_add_field(FixBuilderHandle handle, uint32_t tag, const char* value);
void fix_builder_add_int_field(FixBuilderHandle handle, uint32_t tag, int value);
void fix_builder_add_double_field(FixBuilderHandle handle, uint32_t tag, double value);
int fix_builder_build(FixBuilderHandle handle, char* buffer, int buffer_size);

// Session functions
typedef void* FixSessionHandle;

typedef struct {
    char begin_string[16];
    char sender_comp_id[64];
    char target_comp_id[64];
    int heartbeat_interval;
} FixSessionConfig;

typedef void (*FixMessageCallback)(FixSessionHandle session, FixMessageHandle msg, void* user_data);

FixSessionHandle fix_session_create(const FixSessionConfig* config, FixMessageCallback callback, void* user_data);
void fix_session_destroy(FixSessionHandle handle);
bool fix_session_send(FixSessionHandle handle, const char* msg_type, FixBuilderHandle builder);
bool fix_session_is_logged_on(FixSessionHandle handle);
void fix_session_disconnect(FixSessionHandle handle);

#ifdef __cplusplus
}
#endif

#endif // LX_FIX_BRIDGE_H