cmake_minimum_required(VERSION 3.16)
project(WRF)

set(CMAKE_CXX_STANDARD 11)
set(SOURCES
        src/base/utils/Matrix.cpp
        src/base/utils/Dataset.cpp
        src/base/utils/Metric.cpp
        src/base/CART.cpp
        src/base/RF.cpp
        src/etc/TWRF.cpp src/etc/TWRF.h
        src/etc/WAVE.cpp src/etc/WAVE.h
        src/AWARE.cpp)
set(HEADERS
        include/utils/Matrix.h
        include/utils/Dataset.h
        include/utils/Metric.h
        include/CART.h
        include/RF.h
        include/AWARE.h)

add_executable(WRF main.cpp ${SOURCES} ${HEADERS})

target_include_directories(WRF PUBLIC
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/include/utils
        ${PROJECT_SOURCE_DIR}/src/etc)
