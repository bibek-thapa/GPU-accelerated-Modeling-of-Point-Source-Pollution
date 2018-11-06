#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#define ARRAY_SIZE 10000

extern void pointsource_pollution(float *a, float *b, float *c, int size);

int main (int argc, char *argv[])
{
   float *a = (float *) malloc(ARRAY_SIZE * sizeof(float ));
   float *b = (float *) malloc(ARRAY_SIZE * sizeof(float ));
   float *c = (float *) malloc(ARRAY_SIZE * sizeof(float ));
   int i;

   struct timeval  tv1, tv2;
  	gettimeofday(&tv1, NULL);

    pointsource_pollution(a,b,c,ARRAY_SIZE);
    
  gettimeofday(&tv2, NULL);


  for(int i=0 ; i <ARRAY_SIZE ;i++)
  {
  	printf("%f \t ",a[i]);

  }

  

  printf ("Total time = %f seconds\n",
         (double) (tv2.tv_usec - tv1.tv_usec) / 1000000 +
         (double) (tv2.tv_sec - tv1.tv_sec)); 

  while(1){
    printf("Enter the position where you want to find the ppm");
    printf("\n");
    int x;
    scanf("%d",&x);

     printf("The ppm at  is %f",a[x]);
     printf("\n");

    }         
   free (a);
   free (b);
   free (c);
   
   return 0;
}