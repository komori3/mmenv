cmake_minimum_required(VERSION 3.16)

project(Chokudai001 CXX)

include_directories(../../../include)

add_library(contest_types ./contest_types.cpp)

add_executable(generator ./generator.cpp)
target_compile_features(generator PRIVATE cxx_std_17)
target_link_libraries(generator contest_types)

add_executable(judge ./judge.cpp)
target_compile_features(judge PRIVATE cxx_std_17)
target_link_libraries(judge contest_types)