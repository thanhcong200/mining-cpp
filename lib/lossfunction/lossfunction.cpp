#include "lossfunction.h"
namespace loss{
    double CrossEntropy::GetLoss(const Matrix& prediction, const Matrix& target)
    {
        std::vector<double> predictionVector = prediction.GetColumnVector();
        std::vector<double> targetVector = target.GetColumnVector();
        double sum = 0.0;
        std::vector<double>::iterator tIt = targetVector.begin();
        for (std::vector<double>::iterator pIt = predictionVector.begin(); pIt != predictionVector.end(); ++pIt, ++tIt)
        {
            double value = -*tIt*log(*pIt) - (1 - *tIt)*log(1 - *pIt);
            if (std::isinf(value) || std::isnan(value)) value = std::numeric_limits<int>::max()*1.0;
            sum += value;
        }
        return sum;
    }

    Matrix CrossEntropy::GetDerivative(const Matrix& prediction, const Matrix& target)
    {
        return prediction - target;
    }

    double CategoricalCrossEntropy::GetLoss(Matrix& prediction, Matrix& target){
        std::vector<double> predictionVector = prediction.GetColumnVector();
        std::vector<double> targetVector = target.GetColumnVector();
        double sum = 0.0;
        std::vector<double>::iterator tIt = targetVector.begin();
        for (std::vector<double>::iterator pIt = predictionVector.begin(); pIt != predictionVector.end(); ++pIt, ++tIt)
        {
            double value = -*tIt*log(*pIt);
            if (std::isinf(value) || std::isnan(value)) value = std::numeric_limits<int>::max()*1.0;
            sum += value;
        }
        return sum;
    }

    Matrix CategoricalCrossEntropy::GetDerivative(Matrix& prediction, Matrix& target){
        return prediction - target;
    }
}