cmake_minimum_required(VERSION 3.10)
project(adeus_raspizerow)

# Include FetchContent module
include(FetchContent)

# Fetch cxxopts
FetchContent_Declare(
    cxxopts
    GIT_REPOSITORY https://github.com/jarro2783/cxxopts.git
    GIT_TAG v3.2.0
)
FetchContent_MakeAvailable(cxxopts)

# Add your source files here
set(SOURCES
    main.cpp
)
# Add executable target
add_executable(main ${SOURCES})

# Set C++ standard
set_target_properties(main PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)

include_directories(/usr/include/FLAC)
find_library(FLAC_LIBRARY FLAC PATHS /usr/lib/aarch64-linux-gnu REQUIRED)

# Link against required libraries
find_package(ALSA REQUIRED)
find_package(CURL REQUIRED)
find_package(Threads REQUIRED)

target_link_libraries(main PRIVATE
    ${ALSA_LIBRARIES}
    ${CURL_LIBRARIES}
    Threads::Threads
    cxxopts
    ${FLAC_LIBRARY}
)
