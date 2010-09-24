//kernelPBO.cu (Rob Farber)

#include <stdio.h>
#include <cutil_math.h>
#include <cutil_inline.h>

void checkCUDAError(const char *msg) {
  cudaError_t err = cudaGetLastError();
  if( cudaSuccess != err) {
    fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString( err) ); 
    exit(EXIT_FAILURE); 
  }
} 



__device__ void checkDistance(float lDist, int p, int i, bool * gIntersect, int * gType, int * gIndex, float * gDist){

  if (lDist < *gDist && lDist > 0.0){ //Closest Intersection So Far in Forward Direction of Ray?
    *gType = p; *gIndex = i; *gDist = lDist; *gIntersect = true; //Save Intersection in Global State
  }
}



__device__ void raySphere(int idx, float3 r, float3 o, float4 * spheres,
						  bool * gIntersect, int * gType, int * gIndex, float * gDist) //Ray-Sphere Intersection: r=Ray Direction, o=Ray Origin
{ 
  float3 s = make_float3(spheres[idx].x, spheres[idx].y, spheres[idx].z) - o;  //s=Sphere Center Translated into Coordinate Frame of Ray Origin
  float radius = spheres[idx].w;    //radius=Sphere Radius

  
  //Intersection of Sphere and Line     =       Quadratic Function of Distance
  float A = dot(r,r);                       // Remember This From High School? :

  float B = -2.0 * dot(s,r);                //    A x^2 +     B x +               C  = 0
  float C = dot(s,s) - pow(radius, 2.0f);          // (r'r)x^2 - (2s'r)x + (s's - radius^2) = 0

  float D = B*B - 4*A*C;                     // Precompute Discriminant
  
  if (D > 0.0){                              //Solution Exists only if sqrt(D) is Real (not Imaginary)

    float sign = (C < -0.00001) ? 1 : -1;    //Ray Originates Inside Sphere If C < 0
    float lDist = (-B + sign*sqrt(D))/(2*A); //Solve Quadratic Equation for Distance to Intersection

    checkDistance(lDist,0,idx,gIntersect,gType,gIndex,gDist);             //Is This Closest Intersection So Far?
  }
}


__device__ void rayPlane(int idx, float3 r, float3 o, float2 * planes,
						 bool * gIntersect, int * gType, int * gIndex, float * gDist){ //Ray-Plane Intersection

  int axis = (int) planes[idx].x;            //Determine Orientation of Axis-Aligned Plane

  switch(axis) {
	case 0:
		if (r.x != 0.0){                        //Parallel Ray -> No Intersection
			float lDist = (planes[idx].y - o.x) / r.x; //Solve Linear Equation (rx = p-o)
			checkDistance(lDist,1,idx,gIntersect,gType,gIndex,gDist);
		}
		break;
	case 1:
		if (r.y != 0.0){                        //Parallel Ray -> No Intersection
			float lDist = (planes[idx].y - o.y) / r.y; //Solve Linear Equation (rx = p-o)
			checkDistance(lDist,1,idx,gIntersect,gType,gIndex,gDist);
		}
		break;
	case 2:
		if (r.z != 0.0){                        //Parallel Ray -> No Intersection
			float lDist = (planes[idx].y - o.z) / r.z; //Solve Linear Equation (rx = p-o)
			checkDistance(lDist,1,idx,gIntersect,gType,gIndex,gDist);
		}
		break;
  }

  
}

__device__ void rayObject(int type, int idx, float3 r, float3 o, float4 * spheres, float2 * planes,
						  bool * gIntersect, int * gType, int * gIndex, float * gDist){
	if (type == 0) 
		raySphere(idx,r,o,spheres,gIntersect,gType,gIndex,gDist); 
	else 
		rayPlane(idx,r,o,planes,gIntersect,gType,gIndex,gDist);
}

__device__ void raytrace(float3 ray, float3 origin, int nrTypes, int * nrObjects, float4 * spheres, float2 * planes,
						 bool * gIntersect, int * gType, int * gIndex, float * gDist)
{
  
	*gIntersect = false; //No Intersections Along This Ray Yet
	*gDist = 999999.9;   //Maximum Distance to Any Object

	for (int t = 0; t < nrTypes; t++)
		for (int i = 0; i < nrObjects[t]; i++)
		  rayObject(t,i,ray,origin,spheres,planes,gIntersect,gType,gIndex,gDist);

}


__device__ float3 sphereNormal(int idx, float3 P, float4 * spheres){
  return normalize((P - make_float3(spheres[idx].x, spheres[idx].y, spheres[idx].z))); //Surface Normal (Center to Point)

}

__device__ float3 planeNormal(int idx, float3 P, float3 O, float2 * planes){

  int axis = (int) planes[idx].x;
  float3 N = make_float3(0.0,0.0,0.0);

  switch(axis) {
	case 0:
		N.x = O.x - planes[idx].y;      //Vector From Surface to Light
		break;
	case 1:
		  N.y = O.y - planes[idx].y;
		break;
	case 2:
		  N.z = O.z - planes[idx].y;
		break;
  }
  
  return normalize(N);
}

__device__ float3 surfaceNormal(int type, int index, float3 P, float3 Inside, float2 * planes, float4 * spheres){

  if (type == 0) {return sphereNormal(index,P, spheres);}
  else           {return planeNormal(index,P,Inside,planes);}

}


__device__ float3 reflect(float3 ray, float3 fromPoint, int * gType, int * gIndex, float3 * gPoint, float2 * planes, float4 * spheres){                //Reflect Ray

  float3 N = surfaceNormal(*gType, *gIndex, *gPoint, fromPoint, planes, spheres);  //Surface Normal
  return normalize(ray - (N * (2 * dot(ray,N))));     //Approximation to Reflection

}



__device__ bool gatedSqDist3(float3 a, float3 b, float sqradius, float * gSqDist){ //Gated Squared Distance

  float c = a.x - b.x;          //Efficient When Determining if Thousands of Points
  float d = c*c;                  //Are Within a Radius of a Point (and Most Are Not!)

  if (d > sqradius) return false; //Gate 1 - If this dimension alone is larger than

  c = a.y - b.y;                //         the search radius, no need to continue
  d += c*c;
  if (d > sqradius) return false; //Gate 2

  c = a.z - b.z;
  d += c*c;
  if (d > sqradius) return false; //Gate 3

  *gSqDist = d;      return true ; //Store Squared Distance Itself in Global State

}



__device__ float3 gatherPhotons(float3 p, int type, int id, int ** numPhotons, float3 gOrigin,
								float2 * planes, float4 * spheres, float3 **** photons, float sqRadius,
								float * gSqDist, float exposure){

  float3 energy = make_float3(0.0,0.0,0.0);  
  float3 N = surfaceNormal(type, id, p, gOrigin, planes, spheres);                   //Surface Normal at Current Point

  for (int i = 0; i < numPhotons[type][id]; i++){                    //Photons Which Hit Current Object

    if (gatedSqDist3(p,photons[type][id][i][0],sqRadius, gSqDist)){           //Is Photon Close to Point?
      float weight = max(0.0, -dot(N, photons[type][id][i][1] ));   //Single Photon Diffuse Lighting

      weight *= (1.0 - sqrt(*gSqDist)) / exposure;                    //Weight by Photon-Point Distance
      energy = (energy + (photons[type][id][i][2] * weight)); //Add Photon's Energy to Total

   }} 
  return energy;
}




__device__ float3 computePixelColor(unsigned int x, unsigned int y, float szImg, float3 gOrigin, int nrTypes, int * nrObjects,
									float4 * spheres, float2 * planes, bool * gIntersect, int * gType, int * gIndex, float * gDist, 
									float3 * gPoint, int ** numPhotons, float3 **** photons, float sqRadius, 
									float * gSqDist, float exposure) {
	float3 rgb = make_float3(0.0f, 0.5f, 0.0f);
	float3 ray = make_float3(  x/szImg - 0.5f ,-(y/szImg - 0.5f), 1.0f);	//Convert Pixels to Image Plane Coordinates
																			//Focal Length = 1.0
	raytrace(ray, gOrigin, nrTypes, nrObjects, spheres, planes, gIntersect, gType, gIndex, gDist);

	if(*gIntersect) {
		*gPoint = ray * (*gDist);

		if (*gType == 0 && *gIndex == 1){      //Mirror Surface on This Specific Object
			ray = reflect(ray,gOrigin,gType,gIndex,gPoint,planes,spheres);        //Reflect Ray Off the Surface

			raytrace(ray, *gPoint, nrTypes, nrObjects, spheres, 
					planes, gIntersect, gType, gIndex, gDist);             //Follow the Reflected Ray

			if (gIntersect){ 
				*gPoint =  (ray * (*gDist)) + (*gPoint); 
			}
		} //3D Point of Intersection


		rgb = gatherPhotons(*gPoint,*gType,*gIndex, numPhotons, gOrigin,
								planes, spheres, photons, sqRadius,
								gSqDist, exposure);

	}

	return rgb;
}

//Simple kernel writes changing colors to a uchar4 array
__global__ void kernel(uchar4* pos, unsigned int width, unsigned int height, 
		       float time, int ** numPhotons, float3 **** photons)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int x = index%width;
  unsigned int y = index/width;

  if(index < width*height) {
   
    float3 gOrigin = make_float3(0.0f,0.0f,0.0f);
	int nrTypes = 2; 
	int nrObjects[] = {2,5};          //2 Spheres, 5 Planes
	float4 spheres[] = {make_float4(1.0,0.0,4.0,0.5),make_float4(-0.6,-1.0,4.5,0.5)}; //Sphere Center & Radius
	float2 planes[] = {make_float2(0, 1.5),make_float2(1, -1.5),make_float2(0, -1.5),make_float2(1, 1.5),make_float2(2,5.0)}; //Plane Axis & Distance-to-Origin

	// ----- Raytracing Globals -----
	bool gIntersect = false;       //For Latest Raytracing Call... Was Anything Intersected by the Ray?

	int gType;                        //... Type of the Intersected Object (Sphere or Plane)
	int gIndex;                       //... Index of the Intersected Object (Which Sphere/Plane Was It?)

	float gSqDist, gDist = -1.0;      //... Distance from Ray Origin to Intersection
	float3 gPoint = make_float3(0.0, 0.0, 0.0); //... Point At Which the Ray Intersected the Object


	float sqRadius = 0.7;             //Photon Integration Area (Squared for Efficiency)
	float exposure = 50.0;            //Number of Photons Integrated at Brightest Pixel
	
	


    float3 pixelColor = computePixelColor(x,y,width, gOrigin, nrTypes, nrObjects, spheres, planes, &gIntersect, &gType, &gIndex, &gDist, &gPoint, numPhotons, 
										photons, sqRadius,&gSqDist, exposure);

	unsigned char r = (unsigned char)(pixelColor.x * 255.0f);
    unsigned char g = (unsigned char)(pixelColor.y * 255.0f);
    unsigned char b = (unsigned char)(pixelColor.z * 255.0f);
    
    // Each thread writes one pixel location in the texture (textel)
    pos[index].w = 0;
    pos[index].x = r;
    pos[index].y = g;
    pos[index].z = b;
  }
}



__global__ void render_kernel(uchar4* pos, unsigned int width, unsigned int height, 
							  float time, float3 * pixelData) {

  int index = blockIdx.x * blockDim.x + threadIdx.x;
  // unsigned int x = index%width;
  // unsigned int y = index/width;

  if(index < width*height) {
	
	float3 pixelColor = pixelData[index];

	unsigned char r = (unsigned char)(pixelColor.x);
    unsigned char g = (unsigned char)(pixelColor.y);
    unsigned char b = (unsigned char)(pixelColor.z);
    
    // Each thread writes one pixel location in the texture (textel)
    pos[index].w = 0;
    pos[index].x = r;
    pos[index].y = g;
    pos[index].z = b;
  }

}



extern "C" void launch_render_kernel(uchar4* pos, unsigned int image_width, 
							  unsigned int image_height, float time, float3 pixelData[]) {


    int nThreads=256;
	int totalThreads = image_height * image_width;
	int nBlocks = totalThreads/nThreads; 
	nBlocks += ((totalThreads%nThreads)>0)?1:0;

	float3 * d_pixelData;

	cutilSafeCall(cudaMalloc((void**) &d_pixelData, image_width*image_height*sizeof(float3)));
	cutilSafeCall(cudaMemcpy(d_pixelData, pixelData, image_width*image_height*sizeof(float3),cudaMemcpyHostToDevice));

	render_kernel<<< nBlocks, nThreads>>>(pos, image_width, image_height, time, d_pixelData);

	cudaThreadSynchronize();

	checkCUDAError("kernel failed!");

	cutilSafeCall(cudaFree(d_pixelData));
}


// Wrapper for the __global__ call that sets up the kernel call
extern "C" void launch_kernel(uchar4* pos, unsigned int image_width, 
			      unsigned int image_height, float time, int numPhotons [][5],
				  float *photons[2][5][5000][3])
{
  // execute the kernel
  int nThreads=256;
  int totalThreads = image_height * image_width;
  int nBlocks = totalThreads/nThreads; 
  nBlocks += ((totalThreads%nThreads)>0)?1:0;

  /*
  int ** d_numPhotons;

  float3 **** d_photons;

  cutilSafeCall(cudaMalloc((void**) &d_numPhotons, 2*sizeof(int*)));

  for(int i = 0; i < 2; i++) {
	  cutilSafeCall(cudaMalloc((void**) &d_numPhotons[i], 5*sizeof(int)));
	  cutilSafeCall(cudaMemcpy(d_numPhotons[i], numPhotons[i], 5*sizeof(int), cudaMemcpyHostToDevice));
  }
  
  kernel<<< nBlocks, nThreads>>>(pos, image_width, image_height, time, d_numPhotons, d_photons);
  
  
  for(int i = 0; i < 2; i++) {
	  cutilSafeCall(cudaFree(d_numPhotons[i]));
  }

  cutilSafeCall(cudaFree(d_numPhotons));

  */

  // make certain the kernel has completed 
  cudaThreadSynchronize();

  checkCUDAError("kernel failed!");
}