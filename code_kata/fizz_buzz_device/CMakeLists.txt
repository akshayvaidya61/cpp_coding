cmake_minimum_required(VERSION 3.15)
project(fizz_buzz_device CXX)

add_executable(fizz_buzz_device src/fizz_buzz_device.cpp src/main.cpp)

install(TARGETS fizz_buzz_device DESTINATION "."
        RUNTIME DESTINATION bin
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
        )
