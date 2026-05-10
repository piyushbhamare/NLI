add_library(learned_index_core
    core/types.h
    core/sorted_array.h
    core/sorted_array.cpp
    core/search.cpp
)
add_library(learned_index_baseline
    baseline/linear_model.h
    baseline/linear_model.cpp
    baseline/simple_learned_index.cpp
    baseline/btree.cpp
)
add_library(learned_index_utils
    utils/benchmark.cpp
    utils/statistics.cpp
)
add_executable(learned_index_app main.cpp)
target_link_libraries(learned_index_app learned_index_core learned_index_baseline learned_index_utils)
target_include_directories(learned_index_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(learned_index_baseline PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(learned_index_utils PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
message(STATUS "Source build configured")
