add_library(
    matrix
    matrix/matrix.h
    matrix/matrix.cpp
    lossfunction
    lossfunction/lossfunction.h
    lossfunction/lossfunction.cpp
    activationfunction
    activationfunction/activation.h
    activationfunction/activation.cpp
    layer
    layer/layer.h
    layer/layer.cpp
    model
    model/model.h
    model/model.cpp
)

target_include_directories(matrix PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/matrix")