
#include "myOpenCL.h"

int main(int argc, char ** argv)
{
	std::string strFileName = "test.cl";
	std::string strOpenCLKernalEntry = "hello_kernel";
	cl_mem memObjects[3] = { 0, 0, 0 };
	myOpenCL theOpenCL(strFileName, strOpenCLKernalEntry);
	float result[ARRAY_SIZE];
	float a[ARRAY_SIZE];
	float b[ARRAY_SIZE];
	for (size_t i = 0; i < ARRAY_SIZE; i++)
	{
		a[i] = (float)i;
		b[i] = (float)(i * 2);
	}
	theOpenCL.createMemObjects(memObjects, a, b);

	//建立内核参数
	cl_int errNum = theOpenCL.setKernelParameter(0,memObjects[0]);
	errNum |= theOpenCL.setKernelParameter(1,memObjects[1]);
	errNum |= theOpenCL.setKernelParameter(2,memObjects[2]);

	//使用命令队列使将在设备上执行的内核排队
	size_t globalWorkSize[1] = { ARRAY_SIZE };
	size_t localWorkSize[1] = { 1 };
	errNum = theOpenCL.setKernalQueue(globalWorkSize, localWorkSize);
	//从内核读回结果
	errNum = theOpenCL.readResult(memObjects[2],result);
	for (size_t i = 0; i < ARRAY_SIZE; i++)
	{
		if ( i % 10 == 0 )
		{
			std::cout << std::endl;
		}
		std::cout << result[i] << ",";
	}

	theOpenCL.cleanUp( memObjects);

	return 0;
}