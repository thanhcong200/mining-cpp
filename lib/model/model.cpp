#include "model.h"
#include "../layer/layer.h"
#include <unordered_map>

#define shape(x) std::cout << x.m_Rows <<" "<< x.m_Columns << std::endl
#define print(x) std::cout << x << std::endl

int argmax(std::vector<double> x){
    double max = -1;
    int index = 0;
    for (int i=0; i<x.size(); i++){
        if (x[i] > max){
            max = x[i];
            index = i;
        }
    }
    return index;
}

Model::Model(){}

void Model::Add(Layer::InputLayer inputLayer){
    this->inputLayer = inputLayer;
}

void Model::Add(Layer::HiddenLayer hidden){
    this->hiddenLayer.push_back(hidden);
}

void Model::Add(Layer::OutputLayer outputLayer){
    this->outputLayer = outputLayer;
}

void Model::Initialize(){
    this->outputLayer.Initialize();
    for (int i=0; i<hiddenLayer.size(); i++){
        hiddenLayer[i].Initialize();
    }
}

Matrix Model::Feedforward(Matrix input){
    Matrix x = inputLayer.Forward(input);
    for (int i=0; i<hiddenLayer.size(); i++){
        x = hiddenLayer[i].Forward(x);
    }
    x = outputLayer.Forward(x);
    return x;
}

void Model::SaveMode(std::string fileName){
    std::ofstream outfile;
    outfile.open(fileName, std::ios::binary | std::ios::out);
    for (Layer::HiddenLayer hidden : hiddenLayer){
        hidden.SaveHiddenLayer(outfile);
    }
    this->outputLayer.SaveOutputLayer(outfile);
    outfile.close();
}

void Model::LoadModel(std::string fileName){
    std::ifstream infile;
    infile.open(fileName, std::ios::in | std::ios::binary);
    for (int i=0; i<hiddenLayer.size(); i++){
        hiddenLayer[i] = Layer::HiddenLayer::LoadHiddenLayer(infile);
    }
    this->outputLayer = Layer::OutputLayer::LoadOutputLayer(infile);
    infile.close();
}

float Model::Backpropagation(Matrix inputs, Matrix targets, loss::CategoricalCrossEntropy criterion, float LR){

    std::pair<Matrix, Matrix> deltaOutputWeightBias;
    std::pair<Matrix, Matrix> deltaFirstWeightBias;
    std::unordered_map<unsigned int, std::pair<Matrix, Matrix>> deltaHiddenWeightBias;

    double loss = 0;
    for (int j=0; j<inputs.m_Rows; j++){
        Matrix input(inputs.GetRow(j)); // 1024x1
        Matrix target(targets.GetRow(j)); // 10x1
        Matrix output = Feedforward(input); // 10x1
        loss += criterion.GetLoss(output, target);
        Matrix dL_dZ = criterion.GetDerivative(output, target); // 10x1
        Matrix dZ_dY = outputLayer.ActivationFunction->Derivative(outputLayer.WeightedSum); // 10x1
        Matrix dY_dW = hiddenLayer[hiddenLayer.size()-1].Activation; // 32x1
        Matrix dL_dB = dL_dZ.ElementWise(dZ_dY); // 10x1
        Matrix dL_dW = dL_dB * Matrix::Transpose(dY_dW); // 10x32
        dL_dZ = Matrix::Transpose(outputLayer.WeightMatrix) * dL_dB; // 32x1
        if (j == 0){
            deltaOutputWeightBias.first = dL_dW; // 10x32
            deltaOutputWeightBias.second = dL_dB; // 10x1
        }
        else{
            deltaOutputWeightBias.first += dL_dW;
            deltaOutputWeightBias.second += dL_dB;
        }

        for (int i = hiddenLayer.size() - 1; i>0; i--){
            dZ_dY = hiddenLayer[i].ActivationFunction->Derivative(hiddenLayer[i].WeightedSum); // node x 1
            dY_dW = hiddenLayer[i-1].Activation; //prev_node x 1
            dL_dB = dL_dZ.ElementWise(dZ_dY); // node x 1
            dL_dW = dL_dB * Matrix::Transpose(dY_dW); // node x prev_node
            dL_dZ = Matrix::Transpose(hiddenLayer[i].WeightMatrix) * dL_dB; // prev_node x 1

            if (deltaHiddenWeightBias.find(i) == deltaHiddenWeightBias.end()) {
                deltaHiddenWeightBias[i] = std::make_pair(dL_dW, dL_dB);
            }
            else {
                deltaHiddenWeightBias[i].first += dL_dW;
                deltaHiddenWeightBias[i].second += dL_dB;
            }
        }

        dZ_dY = hiddenLayer[0].ActivationFunction->Derivative(hiddenLayer[0].WeightedSum); // 512x1
        dY_dW = inputLayer.m_Input; // 1024x1
        dL_dB = dL_dZ.ElementWise(dZ_dY); // 512x1
        dL_dW = dL_dB * Matrix::Transpose(dY_dW); // 512x1024
        dL_dZ = Matrix::Transpose(hiddenLayer[0].WeightMatrix) * dL_dB; //1024x1

        if (j == 0){
            deltaFirstWeightBias.first = dL_dW;
            deltaFirstWeightBias.second = dL_dB;
        }
        else{
            deltaFirstWeightBias.first += dL_dW;
            deltaFirstWeightBias.second += dL_dB;
        }
    }

    // Update weight
    outputLayer.WeightMatrix -= LR * deltaOutputWeightBias.first;
    outputLayer.BiasMatrix -= LR * deltaOutputWeightBias.second;
    for (int i = hiddenLayer.size() - 1; i>0; i--){
        hiddenLayer[i].WeightMatrix -= LR * deltaHiddenWeightBias[i].first;
        hiddenLayer[i].BiasMatrix -= LR * deltaHiddenWeightBias[i].second;
    }

    hiddenLayer[0].WeightMatrix -= LR * deltaFirstWeightBias.first;
    hiddenLayer[0].BiasMatrix -= LR * deltaFirstWeightBias.second;

    return loss/inputs.m_Rows;
}

float Model::Eval(Matrix val_dataset, Matrix val_label, std::string dataset){
    int correct = 0;
    int wrong = 0;
    std::ofstream outFile;
    outFile.open("./" + dataset + "/results.txt", std::ios_base::out);
    for (int i=0; i<val_dataset.m_Rows; i++){
        print(i);
        Matrix input = Matrix(val_dataset.GetRow(i));
        Matrix label = Matrix(val_label.GetRow(i));

        Matrix output = Feedforward(input);
        int pred = argmax(output.m_Matrix);
        int gt = argmax(label.m_Matrix);
        outFile << pred << " " << gt<<std::endl;
        if (pred == gt) correct++;
        else wrong++;
    }
    outFile.close();
    return correct*1.0/(correct + wrong);
}

float Model::Valid(Matrix val_dataset, Matrix val_label, loss::CategoricalCrossEntropy criterion){
    float loss = 0;
    for (int i=0; i<val_dataset.m_Rows; i++){
        print(i);
        Matrix input = Matrix(val_dataset.GetRow(i));
        Matrix label = Matrix(val_label.GetRow(i));

        Matrix output = Feedforward(input);
        loss += criterion.GetLoss(output, label);
    }
    return loss/val_dataset.m_Rows;
}