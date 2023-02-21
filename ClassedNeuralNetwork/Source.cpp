#include "NeuralNetwork.h"
#include "LeakyReluLayer.h"

/*
TODO:
1. export and import neural network
2. Runge Kutta 4th order
*/

int main()
{
	float* inputMatrix = nullptr;
	float* outputMatrix = nullptr;
	float* outputDerivativeMatrix = nullptr;
	float* inputDerivativeMatrix = nullptr;
	
	NeuralNetwork neuralNetwork(2, inputMatrix);

	neuralNetwork.AddLayer(new LeakyReluLayer(2));
	neuralNetwork.AddLayer(new LeakyReluLayer(1));
	neuralNetwork.Print();
	neuralNetwork.Initialize(outputMatrix, outputDerivativeMatrix, inputDerivativeMatrix);

	uint32_t index = 0;
	float errors[100] = { 0 };
	float averageError = 0;
	for (uint32_t i = GLOBAL::ITERATIONS; i--;)
	{
		inputMatrix[0] = GLOBAL::random.Rfloat(-10, 10);
		inputMatrix[1] = GLOBAL::random.Rfloat(-10, 10);
		neuralNetwork.ForwardPropagate();
		outputDerivativeMatrix[0] = (inputMatrix[0] + inputMatrix[1]) * 0.5 - outputMatrix[0];
		
		averageError -= errors[index];
		errors[index] = abs(outputDerivativeMatrix[0]);
		averageError += errors[index];
		index -= (++index >= 100) * 100;
		printf("error: %f\n", averageError * 0.01f);
		
		neuralNetwork.BackPropagate(GLOBAL::LEARNING_RATE);
	}
	
	//neuralNetwork.Export("neuralNetwork.txt");

	return 0;
}