#include "NeuralNetwork.h"
#include "LeakyReluLayer.h"

int main()
{
	/**/float* inputMatrix = nullptr;
	float* outputMatrix = nullptr;
	float* outputDerivativeMatrix = nullptr;
	float* inputDerivativeMatrix = nullptr;
	
	NeuralNetwork neuralNetwork(2, inputMatrix);

	neuralNetwork.AddLayer(new LeakyReluLayer(2));
	neuralNetwork.AddLayer(new LeakyReluLayer(1));
	neuralNetwork.Initialize(outputMatrix, outputDerivativeMatrix, inputDerivativeMatrix);

	uint32_t index = 0;
	float errors[100] = { 0 };
	float averageError = 0;
	for (uint32_t i = 1000; i--;)
	{
		inputMatrix[0] = GLOBAL::random.Rfloat(-1, 1);
		inputMatrix[1] = GLOBAL::random.Rfloat(-1, 1);
		neuralNetwork.ForwardPropagate();
		outputDerivativeMatrix[0] = (inputMatrix[0] + inputMatrix[1]) * 0.5 - outputMatrix[0];
		
		averageError -= errors[index];
		errors[index] = abs(outputDerivativeMatrix[0]);
		averageError += errors[index];
		index -= (++index >= 100) * 100;
		printf("error: %f\n", averageError * 0.01f);
		
		neuralNetwork.BackPropagate(0.1f);
	}
	//neuralNetwork.Print();

	return 0;
}