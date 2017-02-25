#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <float.h>
#include <math.h>
#include <string.h>

#include <vector>
#include <limits>
#include <cmath>

#include <cuda_runtime.h>

#include <my_timer.h>
#include <aligned_allocator.h>

#ifndef __RESTRICT
#  define __RESTRICT
#endif

#define NDIM (3)

#ifdef USE_FLOAT
typedef float  ValueType;
#else
typedef double ValueType;
#endif


const ValueType G = 1.0;
const ValueType TINY = std::numeric_limits<ValueType>::epsilon();
const ValueType TINY2 = TINY*TINY;

//#define Enable_ArrayOfStructures
#if defined(Enable_ArrayOfStructures) || defined(__AOS)
#  ifndef Enable_ArrayOfStructures
#    define Enable_ArrayOfStructures
#  endif
   /* Array-of-structures (like) format. */
#  define _index(i,j) (NDIM*(i) + (j))
#else
   /* Structure-of-arrays (like) format. */
#  define _index(i,j) ((i) + (j)*n)
#endif

#define acc_array(i,j) h_acc[ _index((i),(j)) ]
#define pos_array(i,j) h_pos[ _index((i),(j)) ]
#define vel_array(i,j) h_vel[ _index((i),(j)) ]

#define d_acc_array(i, j) d_acc[_index((i),(j))]
#define d_pos_array(i, j) d_pos[_index((i),(j))]
#define d_vel_array(i, j) d_vel[_index((i),(j))]

#define TINY2 (DBL_EPSILON)

#define numBlocks (10)
#define blockSize (1000)

/* Generate a random double between 0,1. */
ValueType frand(void) { return ((ValueType) rand()) / RAND_MAX; }

__global__ void gpu_accel_kernel (ValueType * __RESTRICT d_pos, ValueType * __RESTRICT d_mass, ValueType * __RESTRICT d_acc, const int n)
{   
   int i = blockSize * blockIdx.x + threadIdx.x;
   if (i < n)
   {
   ValueType ax = 0, ay = 0, az = 0;
   const ValueType xi = d_pos_array(i,0);
   const ValueType yi = d_pos_array(i,1);
   const ValueType zi = d_pos_array(i,2);

   for (int j = 0; j < n; ++j)
   {
      /* Position vector from i to j and the distance^2. */
      ValueType rx = d_pos_array(j,0) - xi;
      ValueType ry = d_pos_array(j,1) - yi;
      ValueType rz = d_pos_array(j,2) - zi;
      ValueType dsq = rx*rx + ry*ry + rz*rz + TINY2;
      ValueType m_invR3 = d_mass[j] / (dsq * std::sqrt(dsq));

      ax += rx * m_invR3;
      ay += ry * m_invR3;
      az += rz * m_invR3;
   }

   d_acc_array(i,0) = G * ax;
   d_acc_array(i,1) = G * ay;
   d_acc_array(i,2) = G * az;
   }
}


// Store target data in registers: Compiler "may" do this automatically but
// it often helps with cache efficiency. This can be especially helpfule
// by avoiding repeated writes which are several times slower than reads.
__host__ void accel_gpu (ValueType * __RESTRICT h_pos, ValueType * __RESTRICT h_vel, ValueType * __RESTRICT h_mass, ValueType * __RESTRICT h_acc, const int n)
{
   ValueType *d_pos = NULL;
   ValueType *d_mass = NULL;
   ValueType *d_acc = NULL;

   cudaMalloc(&d_pos, sizeof(ValueType) * n * NDIM);
   cudaMalloc(&d_mass, sizeof(ValueType) * n);
   cudaMalloc(&d_acc, sizeof(ValueType) * n * NDIM);

   cudaMemcpy(d_pos, h_pos, sizeof(ValueType) * n * NDIM, cudaMemcpyHostToDevice);
   cudaMemcpy(d_mass, h_mass, sizeof(ValueType) * n, cudaMemcpyHostToDevice);

   gpu_accel_kernel<<<numBlocks, blockSize>>>(d_pos, d_mass, d_acc, n);

   cudaMemcpy(h_acc, d_acc, sizeof(ValueType) * n * NDIM, cudaMemcpyDeviceToHost);

   cudaFree(d_pos);
   cudaFree(d_mass);
   cudaFree(d_acc);
}

void accel_cpu (ValueType * __RESTRICT h_pos, ValueType * __RESTRICT h_vel, ValueType * __RESTRICT h_mass, ValueType * __RESTRICT h_acc, const int n)
{
   for (int i = 0; i < n; ++i)
   {
      ValueType ax = 0, ay = 0, az = 0;
      const ValueType xi = pos_array(i,0);
      const ValueType yi = pos_array(i,1);
      const ValueType zi = pos_array(i,2);

      for (int j = 0; j < n; ++j)
      {
         /* Position vector from i to j and the distance^2. */
         ValueType rx = pos_array(j,0) - xi;
         ValueType ry = pos_array(j,1) - yi;
         ValueType rz = pos_array(j,2) - zi;
         ValueType dsq = rx*rx + ry*ry + rz*rz + TINY2;
         ValueType m_invR3 = h_mass[j] / (dsq * std::sqrt(dsq));

         ax += rx * m_invR3;
         ay += ry * m_invR3;
         az += rz * m_invR3;
      }

      acc_array(i,0) = G * ax;
      acc_array(i,1) = G * ay;
      acc_array(i,2) = G * az;
   }
}

__global__ void gpu_update_kernel (ValueType d_pos[], ValueType d_vel[], ValueType d_acc[], const int n, ValueType h)
{
   int i = blockSize * blockIdx.x + threadIdx.x;
   if (i < n)
   {
      for (int k = 0; k < NDIM; ++k)
      {
         d_pos_array(i,k) += d_vel_array(i,k)*h + d_acc_array(i,k)*h*h/2;
         d_vel_array(i,k) += d_acc_array(i,k)*h;
      }
   }
}

__host__ void update_gpu (ValueType h_pos[], ValueType h_vel[], ValueType h_mass[], ValueType h_acc[], const int n, ValueType h)
{
   ValueType *d_pos = NULL;
   ValueType *d_vel = NULL;
   ValueType *d_acc = NULL;

   cudaMalloc(&d_pos, sizeof(ValueType) * n * NDIM);
   cudaMalloc(&d_vel, sizeof(ValueType) * n * NDIM);
   cudaMalloc(&d_acc, sizeof(ValueType) * n * NDIM);

   cudaMemcpy(d_pos, h_pos, sizeof(ValueType) * n * NDIM, cudaMemcpyHostToDevice);
   cudaMemcpy(d_vel, h_vel, sizeof(ValueType) * n * NDIM, cudaMemcpyHostToDevice);
   cudaMemcpy(d_acc, h_acc, sizeof(ValueType) * n * NDIM, cudaMemcpyHostToDevice);

   gpu_update_kernel<<<numBlocks, blockSize>>>(d_pos, d_vel, d_acc, n, h);

   cudaMemcpy(h_pos, d_pos, sizeof(ValueType) * n * NDIM, cudaMemcpyDeviceToHost);
   cudaMemcpy(h_vel, d_vel, sizeof(ValueType) * n * NDIM, cudaMemcpyDeviceToHost);
   
   cudaFree(d_pos);
   cudaFree(d_vel);
   cudaFree(d_acc);
}


void update_cpu (ValueType h_pos[], ValueType h_vel[], ValueType h_mass[], ValueType h_acc[], const int n, ValueType h)
{
   for (int i = 0; i < n; ++i)
      for (int k = 0; k < NDIM; ++k)
      {
         pos_array(i,k) += vel_array(i,k)*h + acc_array(i,k)*h*h/2;
         vel_array(i,k) += acc_array(i,k)*h;
      }
}

void output (ValueType h_pos[], ValueType h_vel[], ValueType h_mass[], ValueType h_acc[], const int n, int flnum)
{
   char flname[20];
   sprintf (flname, "pos_%d.out", flnum);
   FILE *fp = fopen(flname,"w");
   if (!fp)
   {
      fprintf(stderr,"Error opening file %s\n", flname);
      exit(-1);
   }

   fwrite (&n, sizeof(int), 1, fp);
   for (int i = 0; i < n; ++i)
   {
      for (int k = 0; k < NDIM; ++k)
      {
         fwrite (&pos_array(i,k), sizeof(ValueType), 1, fp);
      }
      fwrite (&h_mass[i], sizeof(ValueType), 1, fp);
   }

   fclose(fp);
}

__global__ void gpu_search_kernel (ValueType d_pos[], ValueType d_vel[], ValueType d_mass[], ValueType d_acc[], ValueType d_ave[], const int n)
{
   __shared__ ValueType cache[blockSize];
   ValueType minv = 1e10, maxv = 0;
   int i = threadIdx.x;

   for (int k = 0; k < NDIM; ++k)
     cache[i] += (d_vel_array(i + blockSize * blockIdx.x, k) * d_vel_array(i + blockSize * blockIdx.x, k));

   cache[i] = sqrt(cache[i]);

   { 
      maxv = fmax(maxv, cache[i]);
      minv = fmin(minv, cache[i]);
   }

   int j = blockDim.x / 2;
   while (j != 0) {
      if (i < j)
         cache[i] += cache[i + j];
      __syncthreads();
      j /= 2;
   }

   if (i == 0)
      d_ave[blockIdx.x] = cache[0];
}

void search (ValueType h_pos[], ValueType h_vel[], ValueType h_mass[], ValueType h_acc[], const int n)
{
   ValueType minv = 1e10, maxv = 0, ave = 0;
   for (int i = 0; i < n; ++i)
   {
      ValueType vmag = 0;
      for (int k = 0; k < NDIM; ++k)
         vmag += (vel_array(i,k) * vel_array(i,k));

      vmag = sqrt(vmag);

      {
         maxv = std::max(maxv, vmag);
         minv = std::min(minv, vmag);
      }
      ave += vmag;
   }
   printf("min/max/ave velocity = %e, %e, %e\n", minv, maxv, ave / n);
}

void help()
{
   fprintf(stderr,"nbody3 --help|-h --nparticles|-n --nsteps|-s --stepsize|-t\n");
}

__host__ void nbody_gpu (ValueType * __RESTRICT h_pos, ValueType * __RESTRICT h_vel, ValueType * __RESTRICT h_mass, ValueType * __RESTRICT h_acc, ValueType * __RESTRICT h_ave, const int n, ValueType h)
{
   ValueType *d_pos = NULL;
   ValueType *d_vel = NULL;
   ValueType *d_mass = NULL;
   ValueType *d_acc = NULL;
   ValueType *d_ave = NULL;

   cudaMalloc(&d_pos, sizeof(ValueType) * n * NDIM);
   cudaMalloc(&d_vel, sizeof(ValueType) * n * NDIM);
   cudaMalloc(&d_mass, sizeof(ValueType) * n);
   cudaMalloc(&d_acc, sizeof(ValueType) * n * NDIM);
   cudaMalloc(&d_ave, sizeof(ValueType) * numBlocks);

   cudaMemcpy(d_pos, h_pos, sizeof(ValueType) * n * NDIM, cudaMemcpyHostToDevice);
   cudaMemcpy(d_vel, h_vel, sizeof(ValueType) * n * NDIM, cudaMemcpyHostToDevice);
   cudaMemcpy(d_mass, h_mass, sizeof(ValueType) * n, cudaMemcpyHostToDevice);
   cudaMemcpy(d_acc, h_acc, sizeof(ValueType) * n * NDIM, cudaMemcpyHostToDevice);   

   gpu_accel_kernel<<<numBlocks, blockSize>>>(d_pos, d_mass, d_acc, n);
   gpu_update_kernel<<<numBlocks, blockSize>>>(d_pos, d_vel, d_acc, n, h);
   //gpu_search_kernel<<<numBlocks, blockSize>>>(d_pos, d_vel, d_mass, d_acc, d_ave, n);

   cudaMemcpy(h_pos, d_pos, sizeof(ValueType) * n * NDIM, cudaMemcpyDeviceToHost);
   cudaMemcpy(h_vel, d_vel, sizeof(ValueType) * n * NDIM, cudaMemcpyDeviceToHost);
   cudaMemcpy(h_acc, d_acc, sizeof(ValueType) * n * NDIM, cudaMemcpyDeviceToHost);
   cudaMemcpy(h_ave, d_ave, sizeof(ValueType) * numBlocks, cudaMemcpyDeviceToHost);

/*
   // Compute average velocity
   int ave = 0;
   for (int i = 0; i < numBlocks; ++i)
      ave += h_ave[i];
   ave /= n;

   printf("ave velocity = %e\n", ave);
*/
   cudaFree(d_pos);
   cudaFree(d_vel);
   cudaFree(d_mass);
   cudaFree(d_acc);
   cudaFree(d_ave);
}

int main (int argc, char* argv[])
{
   /* Define the number of particles. The default is 1. */
   int num_threads = 1;

   /* Define the number of particles. The default is 100. */
   int n = 100;

   /* Define the number of steps to run. The default is 100. */
   int num_steps = 100;

   /* Pick the timestep size. */
   ValueType dt = 0.01;

   for (int i = 1; i < argc; ++i)
   {
#define check_index(i,str) \
   if ((i) >= argc) \
      { fprintf(stderr,"Missing 2nd argument for %s\n", str); return 1; }

      if ( strcmp(argv[i],"-h") == 0 || strcmp(argv[i],"--help") == 0)
      {
         help();
         return 1;
      }
      else if (strcmp(argv[i],"--nparticles") == 0 || strcmp(argv[i],"-n") == 0)
      {
         check_index(i+1,"--nparticles|-n");
         i++;
         if (isdigit(*argv[i]))
            n = atoi( argv[i] );
      }
      else if (strcmp(argv[i],"--nsteps") == 0 || strcmp(argv[i],"-s") == 0)
      {
         check_index(i+1,"--nsteps|-s");
         i++;
         if (isdigit(*argv[i]))
            num_steps = atoi( argv[i] );
      }
      else if (strcmp(argv[i],"--stepsize") == 0 || strcmp(argv[i],"-t") == 0)
      {
         check_index(i+1,"--stepsize|-t");
         i++;
         if (isdigit(*argv[i]) || *argv[i] == '.')
            dt = atof( argv[i] );
      }
      else
      {
         fprintf(stderr,"Unknown option %s\n", argv[i]);
         help();
         return 1;
      }
   }
   fprintf(stderr,"Number Threads = %d\n", num_threads);
   fprintf(stderr,"Number Objects = %d\n", n);
   fprintf(stderr,"Number Steps   = %d\n", num_steps);
   fprintf(stderr,"Timestep size  = %g\n", dt);
   fprintf(stderr,"Alignment      = %lu bytes\n", Alignment());
   fprintf(stderr,"ValueType      = %s\n", (sizeof(ValueType)==sizeof(double)) ? "double" : "float");
#ifdef Enable_ArrayOfStructures
   fprintf(stderr,"Format         = ArrayOfStructures\n");
#else
   fprintf(stderr,"Format         = StructureOfArrays\n");
#endif

#define _TOSTRING(s) #s
#define TOSTRING(s) _TOSTRING(s)
#ifndef ACC_FUNC
//#  define ACC_FUNC accel_gpu
#  define ACC_FUNC accel_cpu
#endif
#ifndef UPDATE_FUNC
//#  define UPDATE_FUNC update_gpu
#  define UPDATE_FUNC update_cpu
#endif
   fprintf(stderr,"Accel function = %s\n", TOSTRING(ACC_FUNC) );

   ValueType *h_pos = NULL;
   ValueType *h_vel = NULL;
   ValueType *h_acc = NULL;
   ValueType *h_mass = NULL;
   ValueType *h_ave = NULL;

   Allocate(h_pos, n*NDIM);
   Allocate(h_vel, n*NDIM);
   Allocate(h_acc, n*NDIM);
   Allocate(h_mass, n);
   Allocate(h_ave, numBlocks);

   if (1 && n == 2)
   {
      /* Initialize a 2-body problem with large mass ratio and tangential
       * velocity for the small body. */

      pos_array(0,0) = 0.0; pos_array(0,1) = 0.0; pos_array(0,2) = 0.0;
      vel_array(0,0) = 0.0; vel_array(0,1) = 0.0; vel_array(0,2) = 0.0;
      h_mass[0] = 1000.0;

      ValueType vy = std::sqrt(G*h_mass[0]);
      pos_array(1,0) = 1.0; pos_array(1,1) = 0.0; pos_array(1,2) = 0.0;
      vel_array(1,0) = 0.0; vel_array(1,1) =  vy; vel_array(1,2) = 0.0;
      h_mass[1] = 1.0;
   }
   else
   {
      /* Initialize the positions and velocities with random numbers (0,1]. */

      /* 1. Seed the pseudo-random generator. */
      srand(n);

      for (int i = 0; i < n; ++i)
      {
         /* 2. Set some random positions for each object {-1,1}. */
         for (int k = 0; k < NDIM; ++k)
            pos_array(i,k) = 2*(frand() - 0.5);

         /* 3. Set some random velocity (or zero). */
         for (int k = 0; k < NDIM; ++k)
            vel_array(i,k) = 0;
            //vel_array(i,k) = frand();

         /* 4. Set a random mass (> 0). */
         h_mass[i] = frand() + TINY;

         for (int k = 0; k < NDIM; ++k)
            acc_array(i,k) = 0;
      }
   }

   /* Run the step several times. */
   myTimer_t t_start = getTimeStamp();
   double t_accel = 0, t_update = 0, t_search = 0;
   int flnum = 0;
   for (int step = 0; step < num_steps; ++step)
   {
      /* 1. Compute the acceleration on each object. */
      myTimer_t t0 = getTimeStamp();

      //ACC_FUNC( h_pos, h_vel, h_mass, h_acc, n );
      nbody_gpu( h_pos, h_vel, h_mass, h_acc, h_ave, n, dt );

      myTimer_t t1 = getTimeStamp();

      /* 2. Advance the position and velocities. */
      //UPDATE_FUNC( h_pos, h_vel, h_mass, h_acc, n, dt );

      myTimer_t t2 = getTimeStamp();

      /* 3. Find the faster moving object. */
      if (step % 10 == 0)
         search( h_pos, h_vel, h_mass, h_acc, n );

      myTimer_t t3 = getTimeStamp();

      t_accel += getElapsedTime(t0,t1);
      t_update += getElapsedTime(t1,t2);
      t_search += getElapsedTime(t2,t3);

      /* 4. Write positions. */
      if (false && (step % 1 == 0))
      {
         for (int i = 0; i < n; ++i)
         {
            for (int k = 0; k < NDIM; ++k)
               fprintf(stderr,"%f ", pos_array(i,k));
            fprintf(stderr,"%f ", h_mass[i]);
         }
         fprintf(stderr,"\n");
         //output (pos, vel, mass, acc, n, flnum); flnum++;
      }
   }
   double t_calc = getElapsedTime( t_start, getTimeStamp());

   float nkbytes = (float)((size_t)7 * sizeof(ValueType) * (size_t)n) / 1024.0f;
   //printf("Average time = %f (ms) per step with %d elements %.2f KB over %d steps %.3f%%, %.3f%%, %.3f%%\n", t_calc*1000.0/num_steps, n, nkbytes, num_steps, 100*t_accel/t_calc, 100*t_update/t_calc, 100*t_search/t_calc);
   printf("Average time = %f (ms) per step with %d elements %.2f KB over %d steps %f %f %f\n", t_calc*1000.0/num_steps, n, nkbytes, num_steps, t_accel*1000/num_steps, t_update*1000/num_steps, t_search*1000/num_steps);
   /*fclose(fp);*/

   /* Print out the positions (if not too large). */
   if (n < 50)
   {
      for (int i = 0; i < n; ++i)
      {
         for (int k = 0; k < NDIM; ++k)
            fprintf(stderr,"%f ", pos_array(i,k));
         for (int k = 0; k < NDIM; ++k)
            fprintf(stderr,"%f ", vel_array(i,k));

         fprintf(stderr,"%f\n", h_mass[i]);
      }
   }

   Deallocate(h_pos);
   Deallocate(h_vel);
   Deallocate(h_acc);
   Deallocate(h_mass);

   return 0;
}
