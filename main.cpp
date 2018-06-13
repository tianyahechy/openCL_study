
#include "myOpenCL.h"

int main(int argc, char ** argv)
{
	std::string fileName = "test.cl";
	cl_mem memObjects[3] = { 0, 0, 0 };
	myOpenCL theOpenCL;
	cl_context theContext = theOpenCL.createContext();
	cl_device_id  device = NULL;
	cl_command_queue theQueue = theOpenCL.createCommandQueue(theContext, device);
	cl_program theProgram = theOpenCL.createProgram(theContext, device, fileName.c_str());
	//创建opencl内核
	cl_kernel theKernel = clCreateKernel(theProgram, "hello_kernel", NULL);
	if (theKernel == NULL )
	{
		std::cerr << "不能创建内核" << std::endl;
		theOpenCL.cleanUp(theContext, theQueue, theProgram, theKernel, memObjects);
		return 1;
	}
	else
	{
		std::cout << "成功创建内核" << std::endl;
	}

	float result[ARRAY_SIZE];
	float a[ARRAY_SIZE];
	float b[ARRAY_SIZE];
	for (size_t i = 0; i < ARRAY_SIZE; i++)
	{
		a[i] = (float)i;
		b[i] = (float)(i * 2);
	}
	if (!theOpenCL.createMemObjects(theContext, memObjects, a, b))
	{
		std::cout << "创建内存对象失败" << std::endl;
		theOpenCL.cleanUp(theContext, theQueue, theProgram, theKernel, memObjects);
		return 1;
	}
	else
	{
		std::cout << "创建内存对象成功" << std::endl;
	}

	//建立内核参数
	cl_int errNum = clSetKernelArg(theKernel, 0, sizeof(cl_mem), &memObjects[0]);
	errNum |= clSetKernelArg(theKernel, 1, sizeof(cl_mem), &memObjects[1]);
	errNum |= clSetKernelArg(theKernel, 2, sizeof(cl_mem), &memObjects[2]);
	if ( errNum != CL_SUCCESS )
	{
		std::cerr << "错误设置内核参数" << std::endl;
		theOpenCL.cleanUp(theContext, theQueue, theProgram, theKernel, memObjects);
		return 1;
	}
	else
	{
		std::cout << "正确设置内核参数" << std::endl;
	}

	//使用命令队列使将在设备上执行的内核排队
	size_t globalWorkSize[1] = { ARRAY_SIZE };
	size_t localWorkSize[1] = { 1 };
	errNum = clEnqueueNDRangeKernel(theQueue, theKernel, 1, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
	if (errNum != CL_SUCCESS )
	{
		std::cerr << "错误的执行队列内核" << std::endl;
		theOpenCL.cleanUp(theContext, theQueue, theProgram, theKernel, memObjects);
		return 1;
	}
	else
	{
		std::cout << "生成正确的执行队列内核" << std::endl;
	}

	//从内核读回结果
	errNum = clEnqueueReadBuffer(theQueue, memObjects[2], CL_TRUE, 0, ARRAY_SIZE * sizeof(float), result, 0, NULL, NULL);
	if (errNum != CL_SUCCESS)
	{
		std::cerr << "错误读取结果缓冲" << std::endl;
		theOpenCL.cleanUp(theContext, theQueue, theProgram, theKernel, memObjects);
		return 1;
	}
	for (size_t i = 0; i < ARRAY_SIZE; i++)
	{
		if ( i % 10 == 0 )
		{
			std::cout << std::endl;
		}
		std::cout << result[i] << ",";
	}

	theOpenCL.cleanUp(theContext, theQueue, theProgram, theKernel, memObjects);

	return 0;
}