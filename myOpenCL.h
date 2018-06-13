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

	//Ϊcpuƽ̨����������
	cl_context createContext();
	//ѡ���һ�������豸��������һ���������
	cl_command_queue createCommandQueue(cl_context context, cl_device_id & device);
	//�Ӵ��̼����ں�Դ�ļ������͹���һ���������
	cl_program createProgram( const char* fileName);
	//�����ڴ�������豸�ڴ��У��������ں˺���ֱ�ӷ���
	bool createMemObjects( cl_mem memObjects[3], float * a, float *b);
	//�����Դ
	void cleanUp(cl_mem memObjects[3]);
	//�����ں˲���
	cl_int setKernelParameter(int id, cl_mem theData);
	//ʹ���������ʹ�����豸��ִ�е��ں��Ŷ�
	cl_int setKernalQueue(size_t* globalWorkSize, size_t* localWorkSize);
	//���ں˶��ؽ��
	cl_int readResult(cl_mem memObject, float * result);

public:
	//�����豸������ 
	cl_context getContext();
private:
	std::string _strOpenCLFileName; //opencl������ļ�����
	std::string _strOpenCLKernalEntry;//opencl�������
	cl_context _theContext;		//�豸������
	cl_command_queue _commandQueue;//�������
	cl_device_id  _device;	//�豸ID
	cl_program _theProgram; //�������
	cl_kernel _theKernel;//����opencl�ں�
	
};

