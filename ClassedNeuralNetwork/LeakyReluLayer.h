#pragma once
#include "Layer.h"

class LeakyReluLayer : public Layer
{
public:
	uint32_t activationMatrixSize;
	float* activationMatrix;
	
	LeakyReluLayer(uint32_t outputSize)
	{
		activationMatrixSize = outputSize;
		activationMatrix = new float[activationMatrixSize];
	};
	
	~LeakyReluLayer() {};

	void Print() override
	{
		printf("LeakyReluLayer\n");
	}
};