
#include "myOpenCL.h"
#include <vector>

int main(int argc, char ** argv)
{
	std::string strFileName = "test.cl";
	std::string strOpenCLKernalEntry = "hello_kernel";
	int objectSize = 3;
	int numberOfEachObject = ARRAY_SIZE;
	std::vector<cl_mem> memObjectVector;
	memObjectVector.clear();
	memObjectVector.resize(objectSize);
	for (size_t i = 0; i < objectSize; i++)
	{
		memObjectVector[i] = 0;
	}
	myOpenCL theOpenCL(strFileName, strOpenCLKernalEntry);
	std::vector<float> aVector;
	aVector.clear();
	aVector.resize(numberOfEachObject);
	std::vector<float> bVector;
	bVector.clear();
	bVector.resize(numberOfEachObject);
	std::vector<float> resultVector;
	resultVector.clear();
	resultVector.resize(numberOfEachObject);

	for (size_t i = 0; i < numberOfEachObject; i++)
	{
		aVector[i] = (float)i;
		bVector[i] = (float)(i * 2);
	}
	theOpenCL.createMemObjects(&memObjectVector[0], &aVector[0], &bVector[0]);

	//建立内核参数
	for (size_t i = 0; i < objectSize; i++)
	{
		theOpenCL.setKernelParameter(i, memObjectVector[i]);
	}
	//使用命令队列使将在设备上执行的内核排队
	size_t globalWorkSize[1] = { ARRAY_SIZE };
	size_t localWorkSize[1] = { 1 };
	theOpenCL.setKernalQueue(globalWorkSize, localWorkSize);
	//从内核读回结果
	theOpenCL.readResult(memObjectVector[2], &resultVector[0]);
	for (size_t i = 0; i < ARRAY_SIZE; i++)
	{
		if ( i % 10 == 0 )
		{
			std::cout << std::endl;
		}
		std::cout << resultVector[i] << ",";
	}

	theOpenCL.cleanUp(&memObjectVector[0]);

	return 0;
}