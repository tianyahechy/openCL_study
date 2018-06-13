
#include "myOpenCL.h"
#include <vector>

//��д�ڴ�
enum readWrite
{
	readOnly = 0,
	readWrite = 1
};

int main(int argc, char ** argv)
{
	std::string strFileName = "test.cl";
	std::string strOpenCLKernalEntry = "hello_kernel";
	int objectSize = 3;
	int numberOfEachObject = ARRAY_SIZE;
	//��дvector,�����ж��Ƿ�
	myOpenCL theOpenCL(strFileName, strOpenCLKernalEntry);

	std::vector<cl_mem> memObjectVector;
	memObjectVector.clear();
	memObjectVector.resize(objectSize);
	for (size_t i = 0; i < objectSize; i++)
	{
		memObjectVector[i] = 0;
	}
	std::vector<std::vector<float>> computeVector;
	computeVector.clear();
	computeVector.resize(objectSize);
	for (size_t j = 0; j < objectSize; j++)
	{
		computeVector[j].resize(numberOfEachObject);
	}

	for (size_t i = 0; i < numberOfEachObject; i++)
	{
		computeVector[0][i] = (float)i;
		computeVector[1][i] = (float)(i * 2);
	}
	//theOpenCL.createMemObjects(&memObjectVector[0], &computeVector[0][0], &computeVector[1][0]);
	cl_context theContext = theOpenCL.getContext();
	memObjectVector[0] = clCreateBuffer(theContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * ARRAY_SIZE, &computeVector[0][0], NULL);
	memObjectVector[1] = clCreateBuffer(theContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * ARRAY_SIZE, &computeVector[1][0], NULL);
	memObjectVector[2] = clCreateBuffer(theContext, CL_MEM_READ_WRITE, sizeof(float) * ARRAY_SIZE, NULL, NULL);
	//�����ں˲���
	for (size_t i = 0; i < objectSize; i++)
	{
		theOpenCL.setKernelParameter(i, memObjectVector[i]);
	}
	//ʹ���������ʹ�����豸��ִ�е��ں��Ŷ�
	size_t globalWorkSize[1] = { ARRAY_SIZE };
	size_t localWorkSize[1] = { 1 };
	theOpenCL.setKernalQueue(globalWorkSize, localWorkSize);
	//���ں˶��ؽ��
	theOpenCL.readResult(memObjectVector[2], &computeVector[2][0]);
	for (size_t i = 0; i < ARRAY_SIZE; i++)
	{
		if ( i % 10 == 0 )
		{
			std::cout << std::endl;
		}
		std::cout << computeVector[2][i] << ",";
	}

	theOpenCL.cleanUp(&memObjectVector[0]);

	return 0;
}