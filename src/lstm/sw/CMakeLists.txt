cmake_minimum_required(VERSION 3.10)

add_library(SOFT_LSTM STATIC ${CMAKE_SOURCE_DIR}/src/lstm/sw/soft_lstm.cpp)
target_include_directories(SOFT_LSTM PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_include_directories(SOFT_LSTM PUBLIC ${VIVADO_INCLUDE_DIRS})
target_include_directories(SOFT_LSTM PUBLIC ${OpenCv_INCLUDE_DIRS})
target_link_libraries(SOFT_LSTM ${OpenCv_LIBS})

add_library(SOFT_LSTM_SVD STATIC ${CMAKE_SOURCE_DIR}/src/lstm/sw/soft_lstm_svd.cpp)
target_include_directories(SOFT_LSTM_SVD PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_include_directories(SOFT_LSTM_SVD PUBLIC ${VIVADO_INCLUDE_DIRS})
target_include_directories(SOFT_LSTM_SVD PUBLIC ${OpenCv_INCLUDE_DIRS})
target_link_libraries(SOFT_LSTM_SVD ${OpenCv_LIBS})