CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

FIND_PACKAGE(OpenCV REQUIRED)

PROJECT(ProjetCalibration)

ADD_EXECUTABLE(calibration main.cpp)
TARGET_LINK_LIBRARIES(calibration ${OpenCV_LIBS})
