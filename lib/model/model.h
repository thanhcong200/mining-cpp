#pragma one
#include "../layer/layer.h"
#include "../lossfunction/lossfunction.h"

class Model{
    public:
        Layer::InputLayer inputLayer;
        std::vector <Layer::HiddenLayer> hiddenLayer;
        Layer::OutputLayer outputLayer;

        Model();
        void Add(Layer::InputLayer inputLayer); // thêm layer vào model
        void Add(Layer::HiddenLayer hidden);
        void Add(Layer::OutputLayer outputLayer);
        void Initialize(); // khởi tạo tham số
        void SaveMode(std::string fileName); // lưu mô hình
        void LoadModel(std::string fileName); // load mô hình
        Matrix Feedforward(Matrix input); // forward
        float Backpropagation(Matrix input, Matrix target, loss::CategoricalCrossEntropy criterion, float LR); // tính backward
        float Eval(Matrix val_dataset, Matrix val_label, std::string dataset); // evaluation trên tập test
        float Valid(Matrix val_dataset, Matrix val_label, loss::CategoricalCrossEntropy criterion); // tính loss trên tập valid
};