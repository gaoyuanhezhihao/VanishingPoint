find_package(OpenCV REQUIRED)
add_library(vanish vanish.cpp vanish.hpp)
target_link_libraries(vanish ${OpenCV_LIBS})
target_include_directories(vanish PUBLIC  ${CMAKE_CURRENT_SOURCE_DIR})
