#pragma once
#include <iostream>
#include <fstream>
#include <sstream>
#include <time.h>

#ifdef __APPLE__
#include <opencl/cl.h>
#else
#include <CL/cl.h>
#endif

const int ARRAY_SIZE = 10000;
class myOpenCL
{
public:
	myOpenCL();
	~myOpenCL();

	//为cpu平台创建上下文
	cl_context createContext();
	//选择第一个可用设备，并创建一个命令队列
	cl_command_queue createCommandQueue(cl_context context, cl_device_id & device);
	//从磁盘加载内核源文件创建和构建一个程序对象
	cl_program createProgram(cl_context context, cl_device_id device, const char* fileName);
	//创建内存对象，在设备内存中，可以由内核函数直接访问
	bool createMemObjects(cl_context context, cl_mem memObjects[3], float * a, float *b);
	//清空资源
	void cleanUp(cl_context context, cl_command_queue commandQueue, cl_program program, cl_kernel theKernel, cl_mem memObjects[3]);

public:
	//返回设备上下文 
	cl_context getContext();
private:
	cl_context _theContext;		//设备上下文
};

