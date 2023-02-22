#include "NeuralNetwork.h"
#include "LeakyReluLayer.h"

/*
TODO:
1. add Batchsize because right now, runge kutta isn't working that well, perhaps because of the noisy gradient
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
	
	std::ofstream dataFile("data.txt", std::ios::out | std::ios::binary);
	dataFile.write((char*)&GLOBAL::RUNS, 4);
	dataFile.write((char*)&GLOBAL::ITERATIONS, 4);
	
	uint32_t index;
	float errors[100];
	float averageError;
	for (uint32_t run = GLOBAL::RUNS; run--;)
	{
		index = 0;
		memset(errors, 0, 100 * sizeof(float));
		averageError = 0;

		neuralNetwork.Reset();
		
		for (uint32_t i = GLOBAL::ITERATIONS * 10; i--;)
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
			if (i % 10 == 0)
				//printf("error: %f\n", averageError * 0.01f);
				dataFile.write((char*)&averageError, 4);

			neuralNetwork.BackPropagate();
		}
		//printf("-------------------------------------------------------------\n");
	}
	dataFile.close();
	
	//neuralNetwork.Export("neuralNetwork.txt");


	return 0;
}