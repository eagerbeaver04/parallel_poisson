cmake_minimum_required(VERSION 3.10)

project(Project C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED TRUE)

find_package(OpenMP REQUIRED)

add_executable(main_executable src/main.c)
target_link_libraries(main_executable m)

target_compile_options(main_executable PRIVATE
    # -Wall
    # -Wextra
    # -Wpedantic
    # -Wshadow
    # -Wformat=2
    # -Wconversion
    # -O2
    # -g
)

add_executable(main_executable_openmp src/openmp_main.c)
set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
target_link_libraries(main_executable_openmp OpenMP::OpenMP_C m)

target_compile_options(main_executable_openmp PRIVATE
    # -Wall
    # -Wextra
    # -Wpedantic
    # -Wshadow
    # -Wformat=2
    # -Wconversion
    # -O2
    # -g
)
