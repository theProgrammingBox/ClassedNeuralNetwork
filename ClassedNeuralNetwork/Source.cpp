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

	for (uint32_t i = 2; i--;)
	{
		inputMatrix[0] = 0;
		inputMatrix[1] = 1;
		neuralNetwork.ForwardPropagate();
		outputDerivativeMatrix[0] = ((outputMatrix[0] < 0.5f) << 1) - 1.0f;
		neuralNetwork.BackPropagate(0.1f);
		neuralNetwork.Print();
		printf("\n");
	}

	return 0;
}