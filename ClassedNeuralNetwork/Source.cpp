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
	
	std::ofstream file("data.txt", std::ios::out | std::ios::binary);
	
	uint32_t index;
	float errors[100];
	float averageError;
	for (uint32_t run = 10; run--;)
	{
		index = 0;
		memset(errors, 0, 100 * sizeof(float));
		averageError = 0;

		neuralNetwork.Reset();
		
		for (uint32_t i = GLOBAL::ITERATIONS; i--;)
		{
			inputMatrix[0] = GLOBAL::random.Ruint32() & 1;
			inputMatrix[1] = GLOBAL::random.Ruint32() & 1;

			neuralNetwork.ForwardPropagate();
			bool expectedOutput = bool(inputMatrix[0]) ^ bool(inputMatrix[1]);
			outputDerivativeMatrix[0] = float(expectedOutput) - outputMatrix[0];

			averageError -= errors[index];
			errors[index] = abs(outputDerivativeMatrix[0]);
			averageError += errors[index];
			index *= ++index != 100;
			if (i % 1000 == 0)
				printf("error: %f\n", averageError * 0.01f);

			neuralNetwork.BackPropagate();
		}
		printf("-------------------------------------------------------------\n");
	}
	
	//neuralNetwork.Export("neuralNetwork.txt");


	return 0;
}