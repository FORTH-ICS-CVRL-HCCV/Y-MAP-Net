#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include "DataLoader.h"


int main(int argc, char *argv[]) 
{
  fprintf(stderr,"Starting test executable..\n");
  usleep(100);
  test(argc,argv);
}


