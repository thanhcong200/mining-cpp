#include "layer.h"
#define pii std::pair<unsigned int, unsigned int>
#define print(x) std::cout << x << std::endl

namespace Layer{

    Matrix HiddenLayer::Forward(Matrix & input){
        WeightedSum = WeightMatrix*input + BiasMatrix;
        Activation = WeightedSum;
        Activation = ActivationFunction->Function(Activation);
        return Activation;
    }

    HiddenLayer::HiddenLayer(){}
    HiddenLayer::HiddenLayer(unsigned int inputNeurons, unsigned int outputNeurons, activation::Type activationFunction)
        : WeightMatrix(outputNeurons, inputNeurons),
        BiasMatrix(outputNeurons, 1),
        Activation(outputNeurons, 1),
        ActivationFunction(ActivationFunctionFactory::BuildActivationFunction(activationFunction)),
        WeightedSum(outputNeurons, 1)
    {}

    HiddenLayer::HiddenLayer(HiddenLayer && layer) noexcept : WeightMatrix(std::move(layer.WeightMatrix)), BiasMatrix(std::move(layer.BiasMatrix)),
		Activation(std::move(layer.Activation)), WeightedSum(std::move(layer.WeightedSum)), ActivationFunction(std::move(layer.ActivationFunction))
	{}

	HiddenLayer::HiddenLayer(const HiddenLayer & layer) : WeightMatrix(layer.WeightMatrix), BiasMatrix(layer.BiasMatrix),
		Activation(layer.Activation), WeightedSum(layer.WeightedSum), ActivationFunction(layer.ActivationFunction)
	{}

    void HiddenLayer::Initialize(){
        BiasMatrix = Matrix(BiasMatrix.m_Rows, BiasMatrix.m_Columns, 0);
        std::random_device randomDevice;
        std::mt19937 engine(randomDevice());
        std::uniform_real_distribution<double> valueDistribution(0.0, 1.0);
        double factor = 2.0 * sqrt(6.0 / WeightMatrix.GetWidth());
        WeightMatrix.Map([factor, &valueDistribution, &engine](double x)
        {
            return (valueDistribution(engine) - 0.5) * factor;
        });
    }

    void HiddenLayer::SaveHiddenLayer(std::ofstream & outfile) const {
        WeightMatrix.SaveMatrix(outfile);
        BiasMatrix.SaveMatrix(outfile);
        ActivationFunction->SaveActivationFunction(outfile);
    }
    HiddenLayer HiddenLayer::LoadHiddenLayer(std::ifstream & infile) {
        Matrix weightMatrix = Matrix::LoadMatrix(infile);
        Matrix biasMatrix = Matrix::LoadMatrix(infile);
        int activationType;
        infile.read((char*)&activationType, sizeof(activationType));
        activation::Type type = activation::Type(activationType);
        HiddenLayer layer(weightMatrix.GetWidth(), weightMatrix.GetHeight(), type);
        layer.WeightMatrix = std::move(weightMatrix);
        layer.BiasMatrix = std::move(biasMatrix);
        return layer;
    }

    HiddenLayer & HiddenLayer::operator=(HiddenLayer && layer)
	{
		WeightMatrix = std::move(layer.WeightMatrix);
		BiasMatrix = std::move(layer.BiasMatrix);
		Activation = std::move(layer.Activation);
		ActivationFunction = std::move(layer.ActivationFunction);
		WeightedSum = std::move(layer.WeightedSum);
		return *this;
	}

    HiddenLayer & HiddenLayer::operator=(const HiddenLayer & layer) {
        WeightMatrix = std::move(layer.WeightMatrix);
		BiasMatrix = std::move(layer.BiasMatrix);
		Activation = std::move(layer.Activation);
		ActivationFunction = std::move(layer.ActivationFunction);
		WeightedSum = std::move(layer.WeightedSum);
        return *this;
    }

    InputLayer::InputLayer(){}
    InputLayer::InputLayer(const unsigned int input_dims):input_dims(input_dims){
        m_Input = Matrix(input_dims, 1);
    }
    Matrix InputLayer::Forward(Matrix & input){
        this->m_Input = std::move(input);
        return this->m_Input;
    }

    Matrix OutputLayer::Forward(Matrix & input){
        WeightedSum = WeightMatrix*input + BiasMatrix;
        Activation = WeightedSum;
        Activation = ActivationFunction->Function(Activation);
        return Activation;
    }

    OutputLayer::OutputLayer(){}

    OutputLayer::OutputLayer(unsigned int inputNeurons, unsigned int outputNeurons, activation::Type activationFunction)
        : WeightMatrix(outputNeurons, inputNeurons),
        BiasMatrix(outputNeurons, 1),
        Activation(outputNeurons, 1),
        ActivationFunction(ActivationFunctionFactory::BuildActivationFunction(activationFunction)),
        WeightedSum(outputNeurons, 1)
    {}

    OutputLayer::OutputLayer(OutputLayer && layer) noexcept : WeightMatrix(std::move(layer.WeightMatrix)), BiasMatrix(std::move(layer.BiasMatrix)),
		Activation(std::move(layer.Activation)), WeightedSum(std::move(layer.WeightedSum)), ActivationFunction(std::move(layer.ActivationFunction))
	{}

	OutputLayer::OutputLayer(const OutputLayer & layer) : WeightMatrix(layer.WeightMatrix), BiasMatrix(layer.BiasMatrix),
		Activation(layer.Activation), WeightedSum(layer.WeightedSum), ActivationFunction(layer.ActivationFunction)
	{}

    void OutputLayer::Initialize(){
        BiasMatrix = Matrix(BiasMatrix.m_Rows, BiasMatrix.m_Columns, 0);
        std::random_device randomDevice;
        std::mt19937 engine(randomDevice());
        std::uniform_real_distribution<double> valueDistribution(0.0, 1.0);
        double factor = 2.0 * sqrt(6.0 / WeightMatrix.GetWidth());
        WeightMatrix.Map([factor, &valueDistribution, &engine](double x)
        {
            return (valueDistribution(engine) - 0.5) * factor;
        });
    }

    void OutputLayer::SaveOutputLayer(std::ofstream & outfile) const {
        WeightMatrix.SaveMatrix(outfile);
        BiasMatrix.SaveMatrix(outfile);
        ActivationFunction->SaveActivationFunction(outfile);
    }
    OutputLayer OutputLayer::LoadOutputLayer(std::ifstream & infile) {
        Matrix weightMatrix = Matrix::LoadMatrix(infile);
        Matrix biasMatrix = Matrix::LoadMatrix(infile);
        int activationType;
        infile.read((char*)&activationType, sizeof(activationType));
        activation::Type type = activation::Type(activationType);
        OutputLayer layer(weightMatrix.GetWidth(), weightMatrix.GetHeight(), type);
        layer.WeightMatrix = std::move(weightMatrix);
        layer.BiasMatrix = std::move(biasMatrix);
        return layer;
    }

    OutputLayer & OutputLayer::operator=(OutputLayer && layer)
	{
		WeightMatrix = std::move(layer.WeightMatrix);
		BiasMatrix = std::move(layer.BiasMatrix);
		Activation = std::move(layer.Activation);
		ActivationFunction = std::move(layer.ActivationFunction);
		WeightedSum = std::move(layer.WeightedSum);
		return *this;
	}

    OutputLayer & OutputLayer::operator=(const OutputLayer & layer) {
        WeightMatrix = std::move(layer.WeightMatrix);
		BiasMatrix = std::move(layer.BiasMatrix);
		Activation = std::move(layer.Activation);
		ActivationFunction = std::move(layer.ActivationFunction);
		WeightedSum = std::move(layer.WeightedSum);
        return *this;
    }

}