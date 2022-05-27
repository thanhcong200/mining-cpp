#include <iostream>
#include <fstream>
#include "lib/matrix/matrix.cpp"
#include "lib/activationfunction/activation.cpp"
#include "lib/lossfunction/lossfunction.cpp"
#include "lib/layer/layer.cpp"
#include "lib/model/model.cpp"

#define print(x) std::cout << x << std::endl

Model createModel(unsigned int input_dims, unsigned int num_classes){
    Model model;
    model.Add(Layer::InputLayer(input_dims));
    model.Add(Layer::HiddenLayer(input_dims, 512, activation::Type::RELU));
    model.Add(Layer::HiddenLayer(512, 128, activation::Type::RELU));
    model.Add(Layer::HiddenLayer(128, 32, activation::Type::RELU));
    model.Add(Layer::OutputLayer(32, num_classes, activation::Type::SOFTMAX));
    model.Initialize();
    return model;
}

Matrix load_data(std::string data_path, int rows, int cols){
    std::ifstream inFile;
    inFile.open(data_path, std::ios::in);
    if (!inFile) {
        std::cerr << "Unable to open file datafile.txt";
        exit(1);   // call system to stop
    }

    std::vector<double> data(0);
    for (int i = 0; i < rows; i++){
        for(int j = 0; j < cols; j++){
            int pixel;
            inFile >> pixel;
            data.push_back(pixel);
        }
    }
    Matrix mat = Matrix(data, rows, cols);
    inFile.close();
    return mat;
}

std::string dataset = "optical_digits_dataset";

int num_test = 446;
int num_feature = 1024;
int num_classes = 10;

int main(){
    Model model = createModel(num_feature, num_classes);
    model.LoadModel("./" + dataset + "/model_50.bin");
    Matrix mat_test_data = load_data("../data/" + dataset + "/test_data.txt", num_test, num_feature);
    Matrix mat_test_label = load_data("../data/" + dataset + "/test_label.txt", num_test, num_classes);

    float acc = model.Eval(mat_test_data, mat_test_label, dataset);
    print(acc);
}