cmake_minimum_required(VERSION 3.16)
project(parquet_sort CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Arrow C++ 17.0 from pyarrow bundled libraries
set(ARROW_INCLUDE_DIR "$ENV{HOME}/.local/lib/python3.8/site-packages/pyarrow/include"
    CACHE PATH "Arrow include directory")
set(ARROW_LIB_DIR "$ENV{HOME}/.local/lib/python3.8/site-packages/pyarrow"
    CACHE PATH "Arrow library directory")

add_executable(parquet_sort src/parquet_sort.cpp)

target_include_directories(parquet_sort PRIVATE
    ${ARROW_INCLUDE_DIR}
    ${CMAKE_SOURCE_DIR}/src
)

target_link_directories(parquet_sort PRIVATE
    ${ARROW_LIB_DIR}
    ${CMAKE_SOURCE_DIR}/lib
)

target_link_libraries(parquet_sort PRIVATE arrow parquet pthread)

# Set RPATH so the binary finds Arrow .so at runtime
set_target_properties(parquet_sort PROPERTIES
    BUILD_RPATH "${ARROW_LIB_DIR}"
    INSTALL_RPATH "${ARROW_LIB_DIR}"
)

# Release build by default
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -DNDEBUG")

# Shared library for Python multi-threaded parquet writer
add_library(write_mt SHARED src/write_mt.cpp)
target_include_directories(write_mt PRIVATE ${ARROW_INCLUDE_DIR})
target_link_directories(write_mt PRIVATE ${ARROW_LIB_DIR} ${CMAKE_SOURCE_DIR}/lib)
target_link_libraries(write_mt PRIVATE arrow parquet)
set_target_properties(write_mt PROPERTIES
    BUILD_RPATH "${ARROW_LIB_DIR}"
    INSTALL_RPATH "${ARROW_LIB_DIR}"
)
