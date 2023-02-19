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

	inputMatrix[0] = 0;
	inputMatrix[1] = 1;

	neuralNetwork.ForwardPropagate();
	neuralNetwork.Print();

	return 0;
}