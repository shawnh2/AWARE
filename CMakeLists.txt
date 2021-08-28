cmake_minimum_required(VERSION 3.16)
project(WRF)

set(CMAKE_CXX_STANDARD 11)
set(SOURCES
        src/utils/Matrix.cpp src/utils/Matrix.h
        src/utils/Dataset.cpp src/utils/Dataset.h
        src/utils/Metric.cpp src/utils/Metric.h

        src/base/CART.cpp src/base/CART.h
        src/base/RF.cpp src/base/RF.h

        src/etc/TWRF.cpp src/etc/TWRF.h
        src/etc/WAVE.cpp src/etc/WAVE.h
        src/etc/BTA.cpp src/etc/BTA.h
        src/etc/DIRF.cpp src/etc/DIRF.h

        src/AWARE.cpp src/AWARE.h)

add_executable(WRF main.cpp ${SOURCES} ${HEADERS})

target_include_directories(WRF PUBLIC
        ${PROJECT_SOURCE_DIR}/src
        ${PROJECT_SOURCE_DIR}/src/base
        ${PROJECT_SOURCE_DIR}/src/utils
        ${PROJECT_SOURCE_DIR}/src/etc)
