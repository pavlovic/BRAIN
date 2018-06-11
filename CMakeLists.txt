cmake_minimum_required(VERSION 2.8)

project(sbm CXX)

# Set the version numbers
set(SBM_MAJOR 0)
set(SBM_MINOR 00)
set(SBM_PATCH 1)
set(SBM_VERSION ${SBM_MAJOR}.${SBM_MINOR}.${SBM_PATCH})

# configure the version header file 
configure_file (
    "${CMAKE_SOURCE_DIR}/include/SbmVersion.h.in"
    "${CMAKE_BINARY_DIR}/include/SbmVersion.h"
    )
    
# set c++ standard 
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")

# Find armadillo library
find_package(Armadillo REQUIRED)

# include directories
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${ARMADILLO_INCLUDE_DIRS})

# sources
file(GLOB_RECURSE ARMAMIN_SOURCES "src/*.cpp")
file(GLOB_RECURSE ARMAMIN_HEADERS "include/*.h*")
add_executable(sbm ${ARMAMIN_SOURCES} ${ARMAMIN_HEADERS})

# linking
target_link_libraries (sbm ${ARMADILLO_LIBRARIES})

# installation
set(SBM_DIST_DIR ${PROJECT_BINARY_DIR}/${PROJECT_NAME}-${SBM_VERSION})
set(EXECUTABLE_OUTPUT_PATH ${SBM_DIST_DIR}/bin)
install(DIRECTORY SBM_DIST_DIR DESTINATION .)
