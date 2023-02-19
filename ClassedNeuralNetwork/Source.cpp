#include "NeuralNetwork.h"
#include "LeakyReluLayer.h"

int main()
{
	/*float* inputMatrix = nullptr;
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
	neuralNetwork.Print();*/

	float** arrPointer;
	float* arr = new float[2];



	arrPointer = &arr;
	arr[0] = 1;
	arr[1] = 2;


	float* arr2 = new float[2];

	arr2[0] = 3;
	arr2[1] = 4;

	PrintMatrix(*arrPointer, 1, 2, "arrPointer");
	
	arr = arr2;
	delete[] arr2;
	PrintMatrix(*arrPointer, 1, 2, "arrPointer");

	return 0;
}