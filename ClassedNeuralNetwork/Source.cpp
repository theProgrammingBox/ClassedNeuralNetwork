#include "NeuralNetwork.h"
#include "LeakyReluLayer.h"

/*
TODO:
1. visualize
2. export and import neural network
*/

int main()
{
	float* inputMatrix = nullptr;
	float* outputMatrix = nullptr;
	float* outputDerivativeMatrix = nullptr;
	float* inputDerivativeMatrix = nullptr;
	
	NeuralNetwork neuralNetwork(2, inputMatrix);

	neuralNetwork.AddLayer(new LeakyReluLayer(3));
	neuralNetwork.AddLayer(new LeakyReluLayer(1));
	neuralNetwork.Initialize(outputMatrix, outputDerivativeMatrix, inputDerivativeMatrix);

	uint32_t index = 0;
	float errors[100] = { 0 };
	float averageError = 0;
	for (uint32_t i = GLOBAL::ITERATIONS; i--;)
	{
		inputMatrix[0] = GLOBAL::random.Ruint32() & 1;
		inputMatrix[1] = GLOBAL::random.Ruint32() & 1;
		//printf("input: %f %f\n", inputMatrix[0], inputMatrix[1]);
		
		neuralNetwork.ForwardPropagate();
		bool expectedOutput = bool(inputMatrix[0]) ^ bool(inputMatrix[1]);
		outputDerivativeMatrix[0] = float(expectedOutput) - outputMatrix[0];
		/*printf("output: %f\n", outputMatrix[0]);
		printf("expected output: %f\n", float(expectedOutput));
		printf("output derivative: %f\n", outputDerivativeMatrix[0]);*/
		
		averageError -= errors[index];
		errors[index] = abs(outputDerivativeMatrix[0]);
		averageError += errors[index];
		index -= (++index >= 100) * 100;
		printf("error: %f\n", averageError * 0.01f);
		//printf("error: %f\n", abs(outputDerivativeMatrix[0]));
		
		neuralNetwork.BackPropagate();
		//neuralNetwork.Print();
	}
	
	//neuralNetwork.Export("neuralNetwork.txt");


	return 0;
}