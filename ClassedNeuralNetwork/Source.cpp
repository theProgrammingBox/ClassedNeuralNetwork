#include "NeuralNetwork.h"
#include "LeakyReluLayer.h"

int main()
{
	float* inputMatrix = nullptr;
	float* outputMatrix;
	float* outputDerivativeMatrix = nullptr;
	float* inputDerivativeMatrix;
	
	NeuralNetwork neuralNetwork(2, inputMatrix);

	neuralNetwork.AddLayer(new LeakyReluLayer());
	neuralNetwork.AddLayer(new LeakyReluLayer());
	neuralNetwork.Print();

	neuralNetwork.Initialize(outputMatrix, outputDerivativeMatrix, inputDerivativeMatrix);

	return 0;
}