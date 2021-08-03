cmake_minimum_required(VERSION 3.16)
project(WRF)

set(CMAKE_CXX_STANDARD 11)
set(SOURCES
        src/base/utils/Matrix.cpp
        src/base/utils/Dataset.cpp
        src/base/utils/Metric.cpp
        src/base/CART.cpp
        src/base/RF.cpp
        src/etc/TWRF.cpp
        src/etc/WAVE.cpp)
set(HEADERS
        include/base/utils/Matrix.h
        include/base/utils/Dataset.h
        include/base/utils/Metric.h
        include/base/CART.h
        include/base/RF.h
        include/etc/TWRF.h
        include/etc/WAVE.h)

add_executable(WRF main.cpp ${SOURCES} ${HEADERS})

target_include_directories(WRF PUBLIC
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/include/base
        ${PROJECT_SOURCE_DIR}/include/base/utils
        ${PROJECT_SOURCE_DIR}/include/etc)
