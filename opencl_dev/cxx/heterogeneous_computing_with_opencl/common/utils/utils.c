/* System includes */
#include <stdio.h>
#include <stdlib.h>

/* OpenCL includes */
#include <CL/cl.h>

void check(cl_int status, int lineno) {

   if (status != CL_SUCCESS) {
      printf("OpenCL error (%d), line(%d)\n", status, lineno);
      exit(-1);
   }
}

void printCompilerError(cl_program program, cl_device_id device) {
   cl_int status;

   size_t logSize;
   char *log;

   /* Get the log size */
   status = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG,
               0, NULL, &logSize);
   check(status, __LINE__);

   /* Allocate space for the log */
   log = (char*)malloc(logSize);
   if (!log) {
      exit(-1);
   }

   /* Read the log */
   status = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG,
               logSize, log, NULL);
   check(status, __LINE__);

   /* Print the log */
   printf("%s\n", log);
}

char* readFile(const char *filename) {

   FILE *fp;
   char *fileData;
   long fileSize;
   
   /* Open the file */
   fp = fopen(filename, "r");
   if (!fp) {
      printf("Could not open file: %s\n", filename);
      exit(-1);
   }

   /* Determine the file size */
   if (fseek(fp, 0, SEEK_END)) {
      printf("Error reading the file\n");
      exit(-1);
   }
   fileSize = ftell(fp);
   if (fileSize < 0) {
      printf("Error reading the file\n");
      exit(-1);
   }
   if (fseek(fp, 0, SEEK_SET)) {
      printf("Error reading the file\n");
      exit(-1);
   }

   /* Read the contents */
   fileData = (char*)malloc(fileSize + 1);
   if (!fileData) { 
      exit(-1); 
   }
   if (fread(fileData, fileSize, 1, fp) != 1) {
      printf("Error reading the file\n");
      exit(-1);
   }

   /* Terminate the string */
   fileData[fileSize] = '\0';

   /* Close the file */
   if (fclose(fp)) {
      printf("Error closing the file\n");
      exit(-1);
   }

   return fileData;
}
