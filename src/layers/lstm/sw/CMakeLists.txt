cmake_minimum_required(VERSION 3.10)

add_library(SOFT_LSTM STATIC ${CMAKE_SOURCE_DIR}/src/layers/lstm/sw/soft_lstm.cpp)
target_include_directories(SOFT_LSTM PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_include_directories(SOFT_LSTM PUBLIC ${HLS_INCLUDE_DIRS})
target_include_directories(SOFT_LSTM PUBLIC ${OpenCv_INCLUDE_DIRS})
target_link_libraries(SOFT_LSTM ${OpenCv_LIBS})

add_library(SOFT_LSTM_SVD STATIC ${CMAKE_SOURCE_DIR}/src/layers/lstm/sw/soft_lstm_svd.cpp)
target_include_directories(SOFT_LSTM_SVD PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_include_directories(SOFT_LSTM_SVD PUBLIC ${HLS_INCLUDE_DIRS})
target_include_directories(SOFT_LSTM_SVD PUBLIC ${OpenCv_INCLUDE_DIRS})
target_link_libraries(SOFT_LSTM_SVD ${OpenCv_LIBS})
target_link_libraries(SOFT_LSTM_SVD BLAS_UTILS)
target_link_libraries(SOFT_LSTM_SVD ACTIVATION_FUNCTIONS)
