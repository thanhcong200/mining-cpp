#include <iostream>
#include <fstream>
#include "lib/matrix/matrix.cpp"
#include "lib/activationfunction/activation.cpp"
#include "lib/lossfunction/lossfunction.cpp"
#include "lib/layer/layer.cpp"
#include "stdio.h"
#include "string.h"

#define print(x) std::cout << x << std::endl


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
            double pixel;
            inFile >> pixel;
            data.push_back(pixel);
        }
    }
    Matrix mat = Matrix(data);
    mat.m_Rows = rows;
    mat.m_Columns = cols;
    inFile.close();
    return mat;
}

int main(){
    // optical recognition
    Matrix mat_valid_data = load_data("../data/valid_data.txt", 500, 1024);
    Matrix mat_valid_label = load_data("../data/valid_label.txt", 500, 10);

    Matrix mat_train_data = load_data("../data/train_data.txt", 1934, 1024);
    Matrix mat_train_label = load_data("../data/train_label.txt", 1934, 10);

    Matrix mat_test_data = load_data("../data/test_data.txt", 446, 1024);
    Matrix mat_test_label = load_data("../data/test_label.txt", 446, 10);

    // spam classification
    Matrix mat_valid_data = load_data("../data/spam_valid_data.txt", 920, 57);
    Matrix mat_valid_label = load_data("../data/spam_valid_label.txt", 920, 2);

    Matrix mat_train_data = load_data("../data/spam_train_data.txt", 2760, 57);
    Matrix mat_train_label = load_data("../data/spam_train_label.txt", 2760, 2);

    Matrix mat_test_data = load_data("../data/spam_test_data.txt", 921, 57);
    Matrix mat_test_label = load_data("../data/spam_test_label.txt", 921, 2);

    // letter recognition
    Matrix mat_valid_data = load_data("../data/letter_valid_data.txt", 4000, 16);
    Matrix mat_valid_label = load_data("../data/letter_valid_label.txt", 4000, 26);

    Matrix mat_train_data = load_data("../data/letter_train_data.txt", 12000, 16);
    Matrix mat_train_label = load_data("../data/letter_train_label.txt", 12000, 26);

    Matrix mat_test_data = load_data("../data/letter_test_data.txt", 4000, 16);
    Matrix mat_test_label = load_data("../data/letter_test_label.txt", 4000, 26);

    // mushroom 
    Matrix mat_valid_data = load_data("../data/mushroom_valid_data.txt", 1625, 117);
    Matrix mat_valid_label = load_data("../data/mushroom_valid_label.txt", 1625, 2);

    Matrix mat_train_data = load_data("../data/mushroom_train_data.txt", 4874, 117);
    Matrix mat_train_label = load_data("../data/mushroom_train_label.txt", 4874, 2);

    Matrix mat_test_data = load_data("../data/mushroom_test_data.txt", 1625, 117);
    Matrix mat_test_label = load_data("../data/mushroom_test_label.txt", 1625, 2);
}