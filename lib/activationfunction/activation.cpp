#include "activation.h"
#define print(x) std::cout << x << std::endl

namespace activation {
	void ActivationFunction::SaveActivationFunction(std::ofstream & out) const
	{
		Type type = GetType();
		out.write((char*)&type, sizeof(type));
	}

	Matrix ReLu::Function(Matrix& x)
	{
		return x.Map([](double a) { return a >= 0 ? a : 0; });
	}

	Matrix ReLu::Derivative(Matrix& x)
	{
		return x.Map([](double a) { return a >= 0 ? 1 : 0; });
	}

	Type ReLu::GetType() const
	{
		return RELU;
	}

	Matrix Sigmoid::Function(Matrix& x)
	{
		m_Activation = x.Map([](double a) { return 1 / (1 + exp(-a)); });
		return m_Activation;
	}

	Matrix Sigmoid::Derivative(Matrix& x)
	{
		return m_Activation.Map([](double a) { return a * (1 - a); });
	}
	Type Sigmoid::GetType() const
	{
		return SIGMOID;
	}

	Matrix Softmax::Function(Matrix& x)
	{
		double max = 0.0;
		double sum = 0.0;
		std::vector<double> data = x.GetColumnVector();
		for (int i = 0; i < data.size(); i++) if (max < data[i]) max = data[i];
		for (int i = 0; i < data.size(); i++) {
			data[i] = exp(data[i] - max);
			sum += data[i];
		}
		for (int i = 0; i < data.size(); i++) data[i] /= sum;
		m_Activation = Matrix(data, x.m_Rows, x.m_Columns);
		return m_Activation;
	}

	Matrix Softmax::Derivative(Matrix& x)
	{
		return m_Activation.Map([](double a) { return a*(1 - a); });
	}

	Type Softmax::GetType() const
	{
		return SOFTMAX;
	}
}

std::shared_ptr<activation::ActivationFunction> ActivationFunctionFactory::BuildActivationFunction(activation::Type type) {
	switch (type)
	{
	case activation::Type::SIGMOID:
		return std::make_shared<activation::Sigmoid>();
	case activation::Type::RELU:
		return std::make_shared<activation::ReLu>();
	case activation::Type::SOFTMAX:
		return std::make_shared<activation::Softmax>();
	default:
		return nullptr;
	}
}