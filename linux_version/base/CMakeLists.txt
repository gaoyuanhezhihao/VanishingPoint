find_package(Boost 1.54.0 COMPONENTS filesystem regex)

set(SRC base.cpp base.hpp debug.cpp debug.hpp core.hpp core.cpp)
include_directories(${Boost_INCLUDE_DIRS})
add_library(base STATIC ${SRC})
target_link_libraries(base ${Boost_LIBRARIES})
target_link_libraries(base config)
target_include_directories(base PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(base ${OpenCV_LIBS})
add_subdirectory(test)
