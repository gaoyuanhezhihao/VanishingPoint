add_executable(vanish_pt vanish_pt.cpp)
#target_link_libraries(vanish_pt ${OpenCV_LIBS} base lept vanish config)
target_link_libraries(vanish_pt ${OpenCV_LIBS} base vanish config)

add_executable(tagging tagging.cpp)
target_link_libraries(tagging ${OpenCV_LIBS} base vanish config)

add_executable(evaluate evaluate.cpp)
#target_link_libraries(evaluate ${OpenCV_LIBS} base lept vanish config)
target_link_libraries(evaluate ${OpenCV_LIBS} base vanish config)

add_executable(detector_test detector_test.cpp)
target_link_libraries(detector_test ${OpenCV_LIBS} base vanish config)

