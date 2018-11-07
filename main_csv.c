// seqPointSourcePollution 1d
//
// Program to add two vectors using a GPU
#include <stdio.h>
#include <stdlib.h>
#include<string.h>
#define ARRAY_SIZE 10000
#define T 3600000




int main (int argc, char *argv[])
{
   float *a = (float *) malloc(ARRAY_SIZE * sizeof(float));
   float *b = (float *) malloc(ARRAY_SIZE * sizeof(float));
   float *c = (float *) malloc(ARRAY_SIZE * sizeof(float));
   int i,j;
   FILE * fp;

   fp = fopen ("file.csv", "w+");

    for(i=0;i<T;i++)
            {
                if(i%360==0){
                fprintf(fp, "%d,", i);
                }
            }


 fprintf(fp,"\n");
//Initialization of the array
       for (i=0; i < ARRAY_SIZE; i++) {
           a[0]=200.0;
           a[i+1]=0.0;
           b[i]=0.0;
           fprintf(fp,"%f,",a[i]);

       }
        fprintf(fp,"\n");

 for(j = 0; j< T ; j++){

    for(i = 1 ; i < ARRAY_SIZE ; i++)
        {
            b[0]=a[0];
            b[i]=(a[i-1]+a[i+1])/2.0;

        }


    for(i=0; i<ARRAY_SIZE; i++)
        {
            a[i] = b[i];

            if(j%360==0){
            fprintf(fp,"%f,",a[i]);
            }

        }
        if(j%360==0){
         fprintf(fp,"\n");
        }

 }

    fclose(fp);

     printf("\n");

    printf("Enter the position where you want to find the ppm");
    printf("\n");
    int x;
    scanf("%d",&x);

    printf("The ppm at  is %f",a[x]);

   free (a);
   free (b);
   free (c);

   return 0;
}
