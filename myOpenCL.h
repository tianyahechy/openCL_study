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

	//Ϊcpuƽ̨����������
	cl_context createContext();
	//ѡ���һ�������豸��������һ���������
	cl_command_queue createCommandQueue(cl_context context, cl_device_id & device);
	//�Ӵ��̼����ں�Դ�ļ������͹���һ���������
	cl_program createProgram(cl_context context, cl_device_id device, const char* fileName);
	//�����ڴ�������豸�ڴ��У��������ں˺���ֱ�ӷ���
	bool createMemObjects(cl_context context, cl_mem memObjects[3], float * a, float *b);
	//�����Դ
	void cleanUp(cl_context context, cl_command_queue commandQueue, cl_program program, cl_kernel theKernel, cl_mem memObjects[3]);

public:
	//�����豸������ 
	cl_context getContext();
private:
	cl_context _theContext;		//�豸������
};

