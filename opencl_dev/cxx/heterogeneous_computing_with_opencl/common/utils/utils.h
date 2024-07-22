#ifndef __UTILS_H__
#define __UTILS_H__

/* OpenCL includes */
#include <CL/cl.h>

void check(cl_int, int lineno);

void printCompilerError(cl_program program, cl_device_id device);

char* readFile(const char *filename);

#endif
