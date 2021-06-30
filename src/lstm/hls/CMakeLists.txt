cmake_minimum_required(VERSION 3.10)

add_library(LSTM_HARDWARE STATIC ${CMAKE_SOURCE_DIR}/src/lstm/hls/lstm_hardware.cpp)
target_include_directories(LSTM_HARDWARE PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_include_directories(LSTM_HARDWARE PUBLIC ${VIVADO_INCLUDE_DIRS})
target_include_directories(LSTM_HARDWARE PUBLIC ${OpenCv_INCLUDE_DIRS})
target_link_libraries(LSTM_HARDWARE ${OpenCv_LIBS})

add_library(LSTM_SVD STATIC ${CMAKE_SOURCE_DIR}/src/lstm/hls/lstm_svd.cpp)
target_include_directories(LSTM_SVD PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_include_directories(LSTM_SVD PUBLIC ${VIVADO_INCLUDE_DIRS})
target_include_directories(LSTM_SVD PUBLIC ${OpenCv_INCLUDE_DIRS})
target_link_libraries(LSTM_SVD ${OpenCv_LIBS})