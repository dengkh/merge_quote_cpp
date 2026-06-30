cmake_minimum_required(VERSION 3.14)
project(merge_quote_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native")

set(ARROW_INCLUDE_DIR "$ENV{HOME}/.local/lib/python3.8/site-packages/pyarrow/include"
    CACHE PATH "Arrow include directory")
set(ARROW_LIB_DIR "$ENV{HOME}/.local/lib/python3.8/site-packages/pyarrow"
    CACHE PATH "Arrow library directory")

include_directories(${ARROW_INCLUDE_DIR} ${CMAKE_SOURCE_DIR}/src)
link_directories(${ARROW_LIB_DIR} ${CMAKE_SOURCE_DIR}/lib)

add_executable(parquet_sort src/parquet_sort.cpp)
target_link_libraries(parquet_sort arrow parquet pthread stdc++fs)
