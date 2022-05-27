#pragma once
#include "../matrix/matrix.h"
#include <limits>

namespace loss {
    class CrossEntropy 
    {
    public:
        double GetLoss(const Matrix& prediction, const Matrix& target); // tính giá trị loss
        Matrix GetDerivative(const Matrix& prediction, const Matrix& target); // tính đạo hàm 
    };

    class CategoricalCrossEntropy{
        public:
            double GetLoss(Matrix& prediction, Matrix& target);
            Matrix GetDerivative(Matrix& prediction, Matrix& target);
    };

}