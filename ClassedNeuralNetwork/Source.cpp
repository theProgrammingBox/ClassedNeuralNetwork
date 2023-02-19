#include "NeuralNetwork.h"
#include "LeakyReluLayer.h"

int main()
{
	float* inputMatrix = nullptr;
	float* outputMatrix = nullptr;
	float* outputDerivativeMatrix = nullptr;
	float* inputDerivativeMatrix = nullptr;
	
	NeuralNetwork neuralNetwork(2, inputMatrix);

	neuralNetwork.AddLayer(new LeakyReluLayer(2));
	neuralNetwork.AddLayer(new LeakyReluLayer(1));
	neuralNetwork.Initialize(outputMatrix, outputDerivativeMatrix, inputDerivativeMatrix);
	neuralNetwork.Print();

	return 0;
}