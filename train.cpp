#include <iostream>
#include <fstream>
#include "lib/matrix/matrix.cpp"
#include "lib/activationfunction/activation.cpp"
#include "lib/lossfunction/lossfunction.cpp"
#include "lib/layer/layer.cpp"
#include "lib/model/model.cpp"

#define shape(x) std::cout << x.m_Rows <<" "<< x.m_Columns << std::endl
#define print(x) std::cout << x << std::endl
#define pii std::pair<unsigned int, unsigned int>
#define vb std::vector<double>

Model createModel(unsigned int input_dims, unsigned int num_classes){
    Model model;
    // optical digit
    model.Add(Layer::InputLayer(input_dims));
    model.Add(Layer::HiddenLayer(input_dims, 512, activation::Type::RELU));
    model.Add(Layer::HiddenLayer(512, 128, activation::Type::RELU));
    model.Add(Layer::HiddenLayer(128, 32, activation::Type::RELU));
    model.Add(Layer::OutputLayer(32, num_classes, activation::Type::SIGMOID));
    
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
int num_train = 1934;
int num_val = 500;
int num_feature = 1024;
int num_classes = 10;
int EPOCHS = 50;
float LR = 0.001;
int BATCH_SIZE = 64;

int main(){
    Model model = createModel(num_feature, num_classes);
    loss::CategoricalCrossEntropy criterion;
    Matrix mat_train_data = load_data("./data/" + dataset + "/train_data.txt", num_train, num_feature);
    Matrix mat_train_label = load_data("./data/" + dataset + "/train_label.txt", num_train, num_classes);
    Matrix mat_valid_data = load_data("./data/" + dataset + "/valid_data.txt", num_val, num_feature);
    Matrix mat_valid_label = load_data("./data/" + dataset + "/valid_label.txt", num_val, num_classes);

    std::vector<int> index;
    for (int i=0; i<num_train; i++){
        index.push_back(i);
    }
    std::ofstream outFile;
    outFile.open("loss.txt", std::ios_base::out);
    std::ofstream logFile;
    logFile.open("log.txt", std::ios_base::out);
    for (int e=1; e<=EPOCHS; e++){
        std::random_shuffle ( index.begin(), index.end() );
        if (e % 15 == 0){
            LR /= 10;
        }

        std::cout<<"EPOCH: "<<e<<": ";
        std::cout << "[";
        outFile << "EPOCH: "<<e<<" <<<<<<<<<" <<std::endl;
        logFile << "EPOCH: "<<e<<" <<<<<<<<<" <<std::endl;

        Matrix input;
        Matrix label;

        float total = 0;
        int c = 0;
        for (int idx = 0; idx<index.size(); idx++){
            int i = index[idx];
            if (idx == 0){
                input = Matrix(mat_train_data.GetRow(i), 1, num_feature);
                label = Matrix(mat_train_label.GetRow(i), 1, num_classes);
            }
            else if (idx % BATCH_SIZE == 0 || idx == index.size() - 1){
                c++;
                float loss = model.Backpropagation(input, label, criterion, LR);
                total += loss;
                outFile << loss <<std::endl;
                std::cout << "#";
                std::cout.flush();
                input = Matrix::Transpose(Matrix(mat_train_data.GetRow(i)));
                label = Matrix::Transpose(Matrix(mat_train_label.GetRow(i)));
                // break;
            }
            else{
                input.AddRow(mat_train_data.GetRow(i));
                label.AddRow(mat_train_label.GetRow(i));
            }
        }
        std::cout << "]\n";
        float valid_loss = model.Valid(mat_valid_data, mat_valid_label, criterion);
        logFile << total/c <<" "<<valid_loss<<std::endl;
        model.SaveMode(std::to_string(e) + ".bin");
    }
    outFile.close();
    logFile.close();

    return 0;
}