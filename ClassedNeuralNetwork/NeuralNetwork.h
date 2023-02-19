#pragma once
#include "Layer.h"

class NeuralNetwork
{
public:
	uint32_t inputMatrixSize;
	float* inputMatrix;
	float* outputDerivativeMatrix;
	
	uint32_t staticMatrixSize;
	uint32_t dynamicMatrixSize;
	float* staticMatrix;
	float* dynamicParamMatrix;
	float* dynamicParamDerivitiveMatrix;
	float* currentDynamicParamDerivitiveLocation;
	
	std::vector<Layer*> layers;

	NeuralNetwork(uint32_t inputMatrixSize, float* inputMatrix)
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

	void Initialize(float* outputMatrix, float* outputDerivativeMatrix, float* inputDerivativeMatrix)
	{
		std::vector<StaticMatrixInfo> staticParams;
		std::vector<DynamicMatrixInfo> dynamicParams;

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
			matrixInfo.displacement = dynamicMatrixSize;
			dynamicMatrixSize += matrixInfo.matrixSize;
		}
		
		staticMatrix = new float[staticMatrixSize];
		dynamicParamMatrix = new float[dynamicMatrixSize];
		dynamicParamDerivitiveMatrix = new float[dynamicMatrixSize << 2];

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
		
		cpuGenerateUniform(dynamicParamMatrix, dynamicMatrixSize, -1, 1);
		PrintMatrix(dynamicParamMatrix, 1, dynamicMatrixSize, "Dynamic Matrix");

		/*for (Layer* layer : layers)
			layer->AssignDynamicDerivativeMatrix*/
	}

	void Print()
	{
		for (auto& layer : layers)
			layer->Print();
	}
};