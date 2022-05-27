#pragma once
#include "../activationfunction/activation.h"
#include <memory>


namespace Layer{
    class HiddenLayer{
        public:
            Matrix WeightMatrix; // ma trận trọn số
            Matrix BiasMatrix; // ma trậN bias
            std::shared_ptr<activation::ActivationFunction> ActivationFunction; // hàm kích hoạt
            Matrix Activation; // ma trận sau khi đưa qua hàm kích hoạt
            Matrix WeightedSum; // ma trận trước khi đưa qua activation Wx + B

            HiddenLayer(); // hàm khởi tạo
            HiddenLayer(unsigned int inputNeurons, unsigned int outputNeurons, activation::Type activationFunction); // hàm khởi tạo
            void Initialize(); // khởI tạo trọng số
            void SaveHiddenLayer(std::ofstream& outfile) const; // lưu layer
            static HiddenLayer LoadHiddenLayer(std::ifstream& infile); // load
            HiddenLayer(HiddenLayer&& layer) noexcept; // hàm khỏi tạo
            HiddenLayer(const HiddenLayer& layer); // hàm khỏi tạo
            HiddenLayer& operator=(HiddenLayer&& layer); // ghi đè toán tử gán
            HiddenLayer& operator=(const HiddenLayer& matrix); // ghi đè toán tử gán
            Matrix Forward(Matrix & input); // tính forward
    };

    class InputLayer{
        public:
            unsigned int input_dims;
            Matrix m_Input;
            
            InputLayer(); // hàm khỏi tạo
            InputLayer(const unsigned int input_dims); //hàm khỏi tạo
            Matrix Forward(Matrix & input); // tính forward

    };

    class OutputLayer{ // tương tự hidden layer
        public:
            Matrix WeightMatrix;
            Matrix BiasMatrix;
            std::shared_ptr<activation::ActivationFunction> ActivationFunction;
            Matrix Activation;
            Matrix WeightedSum;

            OutputLayer();
            OutputLayer(unsigned int inputNeurons, unsigned int outputNeurons, activation::Type activationFunction);
            void Initialize();
            void SaveOutputLayer(std::ofstream& outfile) const;
            static OutputLayer LoadOutputLayer(std::ifstream& infile);
            OutputLayer(OutputLayer&& layer) noexcept;
            OutputLayer(const OutputLayer& layer);
            OutputLayer& operator=(OutputLayer&& layer);
            OutputLayer& operator=(const OutputLayer& matrix);
            Matrix Forward(Matrix & input);
    };

}