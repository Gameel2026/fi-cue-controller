#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
typedef struct { float on, off, gate; int32_t k, hold, use_arrest; float drop, walk; int32_t arrest_hold; } ctrl_params;
int run(const float*,int,float*,uint8_t*,const ctrl_params*);
int main(int c,char**v){FILE*f=fopen(v[1],"rb");long n;if(fread(&n,8,1,f)!=1)return 1;float*x=malloc(12*n);if(fread(x,4,3*n,f)!=(size_t)(3*n))return 1;fclose(f);
ctrl_params p; FILE*q=fopen(v[2],"rb"); if(fread(&p,sizeof p,1,q)!=1)return 1; fclose(q);
float*o=malloc(8*(n/16+2));uint8_t*cu=malloc(n/16+2);int h=run(x,(int)n,o,cu,&p);FILE*g=fopen(v[3],"wb");fwrite(cu,1,h,g);fclose(g);printf("decisions %d\n",h);return 0;}
