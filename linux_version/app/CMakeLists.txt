add_executable(vanish_pt main.cpp)
target_link_libraries(vanish_pt ${OpenCV_LIBS} lept vanish config)
