cmake_minimum_required(VERSION 3.24)
project(seam_carving)

set(CMAKE_CXX_STANDARD 17)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

include(FetchContent)
FetchContent_Declare(
        argparse
        GIT_REPOSITORY https://github.com/p-ranav/argparse.git
)
FetchContent_MakeAvailable(argparse)

add_executable(seam_carving main.cpp)
target_link_libraries(seam_carving ${OpenCV_LIBS})

SET(MAIN_EXEC main)

FOREACH(FILE ${MAIN_EXEC})
    add_executable(${FILE} ${FILE}.cpp)
    target_link_libraries(${FILE} ${OpenCV_LIBS} argparse)
ENDFOREACH(FILE)
