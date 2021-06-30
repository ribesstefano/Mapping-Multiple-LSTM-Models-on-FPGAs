cmake_minimum_required(VERSION 3.10)
add_subdirectory(hls)
add_subdirectory(sw)

add_library(LSTM_DATA_HANDLER STATIC ${CMAKE_SOURCE_DIR}/src/lstm/lstm_data_handler.cpp)
target_include_directories(LSTM_DATA_HANDLER PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_include_directories(LSTM_DATA_HANDLER PUBLIC ${VIVADO_INCLUDE_DIRS})
target_include_directories(LSTM_DATA_HANDLER PUBLIC ${OpenCv_INCLUDE_DIRS})
target_link_libraries(LSTM_DATA_HANDLER ${OpenCv_LIBS})