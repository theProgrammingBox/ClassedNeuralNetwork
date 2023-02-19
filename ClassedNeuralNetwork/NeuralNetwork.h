#pragma once
#include "Layer.h"

class NeuralNetwork
{
public:
	uint32_t inputMatrixSize;
	float* inputMatrix;
	float* outputMatrix;
	float* outputDerivativeMatrix;
	float* inputDerivativeMatrix;
	
	uint32_t staticMatrixSize;
	uint32_t dynamicMatrixSize;
	uint32_t dynamicMatrixDerivitiveSize;
	uint32_t dynamicParamDerivitiveDisplacement;
	float* staticMatrix;
	float* dynamicParamMatrix;
	float* dynamicParamDerivitiveMatrix;
	float* currentDynamicParamDerivitiveLocation;
	
	std::vector<Layer*> layers;

	NeuralNetwork(uint32_t inputMatrixSize, float*& inputMatrix)
	{
		this->inputMatrixSize = inputMatrixSize;
		this->inputMatrix = new float[inputMatrixSize];
		inputMatrix = this->inputMatrix;
	}
	
	~NeuralNetwork()
	{
		delete[] inputMatrix;
		delete[] outputDerivativeMatrix;
		for (Layer* layer : layers)
			delete layer;

		delete[] staticMatrix;
		delete[] dynamicParamMatrix;
		delete[] dynamicParamDerivitiveMatrix;
	}

	void AddLayer(Layer* layer)
	{
		layers.emplace_back(layer);
	}

	void Initialize(float*& outputMatrix, float*& outputDerivativeMatrix, float*& inputDerivativeMatrix)
	{
		std::vector<StaticMatrixInfo> staticParams;
		std::vector<DynamicMatrixInfo> dynamicParams;

		this->outputDerivativeMatrix = new float[layers.back()->GetOutputMatrixSize()];
		outputDerivativeMatrix = this->outputDerivativeMatrix;

		for (uint32_t i = layers.size() - 1; i--;)
		{
			layers[i + 1]->AssignInputMatrixSize(layers[i]->GetOutputMatrixSize());
			layers[i + 1]->LoadLayerSpecs(
				staticParams,
				dynamicParams
			);
		}
		layers[0]->AssignInputMatrixSize(inputMatrixSize);
		layers[0]->LoadLayerSpecs(
			staticParams,
			dynamicParams
		);

		staticMatrixSize = 0;
		for (StaticMatrixInfo& matrixInfo : staticParams)
			staticMatrixSize += matrixInfo.matrixSize;
		
		dynamicMatrixSize = 0;
		for (DynamicMatrixInfo& matrixInfo : dynamicParams)
		{
			*matrixInfo.displacement = dynamicMatrixSize;
			dynamicMatrixSize += matrixInfo.matrixSize;
		}
		
		dynamicMatrixDerivitiveSize = dynamicMatrixSize << 2;
		dynamicParamDerivitiveDisplacement = 0;
		staticMatrix = new float[staticMatrixSize];
		dynamicParamMatrix = new float[dynamicMatrixSize];
		dynamicParamDerivitiveMatrix = new float[dynamicMatrixDerivitiveSize];
		currentDynamicParamDerivitiveLocation = dynamicParamDerivitiveMatrix;
		ResetDynamicParamDerivitiveMatrix();
		
		cpuGenerateUniform(dynamicParamMatrix, dynamicMatrixSize, -1, 1);

		float* staticMatrixIndex = staticMatrix;
		for (StaticMatrixInfo& matrixInfo : staticParams)
		{
			*matrixInfo.matrix = staticMatrixIndex;
			staticMatrixIndex += matrixInfo.matrixSize;
		}

		float* dynamicMatrixLocation = dynamicParamMatrix;
		for (DynamicMatrixInfo& matrixInfo : dynamicParams)
		{
			*matrixInfo.matrix = dynamicMatrixLocation;
			dynamicMatrixLocation += matrixInfo.matrixSize;
		}

		outputMatrix = layers.back()->GetOutputMatrix();
		inputDerivativeMatrix = layers[0]->GetInputDerivativeMatrix();

		layers.back()->AssignOutputDerivativeMatrix(outputDerivativeMatrix);
		layers.back()->dynamicDerivativeMatrixPointer = &currentDynamicParamDerivitiveLocation;
		for (uint32_t i = layers.size() - 1; i--;)
		{
			layers[i + 1]->AssignInputMatrix(layers[i]->GetOutputMatrix());
			layers[i]->dynamicDerivativeMatrixPointer = &currentDynamicParamDerivitiveLocation;
			layers[i]->AssignOutputDerivativeMatrix(layers[i + 1]->GetInputDerivativeMatrix());
		}
		layers[0]->AssignInputMatrix(inputMatrix);
	}

	void ForwardPropagate()
	{
		//ResetDynamicParamDerivitiveMatrix();
		for (auto& layer : layers)
			layer->ForwardPropagate();
	}

	void BackPropagate(float dt)
	{
		for (uint32_t i = layers.size(); i--;)
			layers[i]->BackPropagate();
		cpuSaxpy(dynamicMatrixSize, &dt, dynamicParamDerivitiveMatrix, 1, dynamicParamMatrix, 1);
	}

	void Print()
	{
		//memset(dynamicParamDerivitiveMatrix, 0, dynamicMatrixDerivitiveSize * sizeof(float));
		printf("------------------------------------\n");
		PrintMatrix(dynamicParamDerivitiveMatrix, 1, dynamicMatrixDerivitiveSize, "Dynamic Param Derivitive Matrix");
		printf("------------------------------------\n");
		
		for (auto& layer : layers)
			layer->Print();
		printf("\n\n");
	}

private:
	void ResetDynamicParamDerivitiveMatrix()
	{
		memset(dynamicParamDerivitiveMatrix, 0, dynamicMatrixDerivitiveSize * sizeof(float));
	}
};