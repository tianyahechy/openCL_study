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
	myOpenCL(std::string strOpenCLFileName, std::string strOpenCLKernalEntry);
	~myOpenCL();

	//为cpu平台创建上下文
	cl_context createContext();
	//选择第一个可用设备，并创建一个命令队列
	cl_command_queue createCommandQueue(cl_context context, cl_device_id & device);
	//从磁盘加载内核源文件创建和构建一个程序对象
	cl_program createProgram( const char* fileName);
	//创建内存对象，在设备内存中，可以由内核函数直接访问
	bool createMemObjects( cl_mem memObjects[3], float * a, float *b);
	//清空资源
	void cleanUp(cl_mem memObjects[3]);
	//建立内核参数
	cl_int setKernelParameter(int id, cl_mem theData);
	//使用命令队列使将在设备上执行的内核排队
	cl_int setKernalQueue(size_t* globalWorkSize, size_t* localWorkSize);
	//从内核读回结果
	cl_int readResult(cl_mem memObject, float * result);

public:
	//返回设备上下文 
	cl_context getContext();
private:
	std::string _strOpenCLFileName; //opencl处理的文件名称
	std::string _strOpenCLKernalEntry;//opencl入口名称
	cl_context _theContext;		//设备上下文
	cl_command_queue _commandQueue;//命令队列
	cl_device_id  _device;	//设备ID
	cl_program _theProgram; //程序对象
	cl_kernel _theKernel;//创建opencl内核
	
};

