cmake_minimum_required(VERSION 3.10)
project(lxdex VERSION 1.0.0 LANGUAGES C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Options
option(LXDEX_BUILD_EXAMPLES "Build examples" ON)
option(LXDEX_BUILD_TESTS "Build tests" OFF)

# Find libwebsockets
find_path(LWS_INCLUDE_DIR libwebsockets.h
    HINTS /usr/include /usr/local/include /opt/homebrew/include
    PATH_SUFFIXES libwebsockets)
find_library(LWS_LIBRARY NAMES websockets libwebsockets
    HINTS /usr/lib /usr/local/lib /opt/homebrew/lib)

if(NOT LWS_INCLUDE_DIR OR NOT LWS_LIBRARY)
    message(FATAL_ERROR "libwebsockets not found. Install with: brew install libwebsockets (macOS) or apt install libwebsockets-dev (Linux)")
endif()

set(LIBWEBSOCKETS_INCLUDE_DIRS ${LWS_INCLUDE_DIR})
set(LIBWEBSOCKETS_LIBRARIES ${LWS_LIBRARY})

message(STATUS "libwebsockets include: ${LIBWEBSOCKETS_INCLUDE_DIRS}")
message(STATUS "libwebsockets library: ${LIBWEBSOCKETS_LIBRARIES}")

# Find OpenSSL (required by libwebsockets)
if(APPLE)
    set(OPENSSL_ROOT_DIR /opt/homebrew/opt/openssl@3)
endif()
find_package(OpenSSL REQUIRED)
message(STATUS "OpenSSL include: ${OPENSSL_INCLUDE_DIR}")

# Platform-specific settings
if(APPLE)
    # macOS
    set(PLATFORM_LIBS "-framework Security -framework CoreFoundation")
elseif(UNIX)
    # Linux
    set(PLATFORM_LIBS pthread)
endif()

# Library sources
set(LXDEX_SOURCES
    src/client.c
    src/orderbook.c
    src/json.c
)

# Create static library
add_library(lxdex STATIC ${LXDEX_SOURCES})

target_include_directories(lxdex
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${LIBWEBSOCKETS_INCLUDE_DIRS}
        ${OPENSSL_INCLUDE_DIR}
)

target_link_libraries(lxdex
    PRIVATE
        ${LIBWEBSOCKETS_LIBRARIES}
        OpenSSL::SSL
        OpenSSL::Crypto
        ${PLATFORM_LIBS}
)

target_compile_options(lxdex PRIVATE ${LIBWEBSOCKETS_CFLAGS_OTHER})

# Shared library
add_library(lxdex_shared SHARED ${LXDEX_SOURCES})
set_target_properties(lxdex_shared PROPERTIES OUTPUT_NAME lxdex)

target_include_directories(lxdex_shared
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${LIBWEBSOCKETS_INCLUDE_DIRS}
        ${OPENSSL_INCLUDE_DIR}
)

target_link_libraries(lxdex_shared
    PRIVATE
        ${LIBWEBSOCKETS_LIBRARIES}
        OpenSSL::SSL
        OpenSSL::Crypto
        ${PLATFORM_LIBS}
)

target_compile_options(lxdex_shared PRIVATE ${LIBWEBSOCKETS_CFLAGS_OTHER})

# Examples
if(LXDEX_BUILD_EXAMPLES)
    add_executable(basic examples/basic.c)
    target_link_libraries(basic PRIVATE lxdex ${LIBWEBSOCKETS_LIBRARIES} OpenSSL::SSL OpenSSL::Crypto ${PLATFORM_LIBS})
    target_include_directories(basic PRIVATE ${LIBWEBSOCKETS_INCLUDE_DIRS} ${OPENSSL_INCLUDE_DIR})
endif()

# Install rules
include(GNUInstallDirs)

install(TARGETS lxdex lxdex_shared
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

install(FILES include/lxdex.h
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

# pkg-config
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/lxdex.pc.in
    ${CMAKE_CURRENT_BINARY_DIR}/lxdex.pc
    @ONLY
)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/lxdex.pc
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/pkgconfig
)
