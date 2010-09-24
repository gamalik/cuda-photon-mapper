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


	// ----- Scene Description -----
	__device__ int szImg = 512;                  //Image Size
	__device__ int nrTypes = 2;                  //2 Object Types (Sphere = 0, Plane = 1)
	__device__ int nrObjects [] = {2,5};          //2 Spheres, 5 Planes
	__device__ float gAmbient = 0.1;             //Ambient Lighting
	__device__ float3 gOrigin; // = make_float3(0.0,0.0,0.0);  //World Origin for Convenient Re-Use Below (Constant)
	__device__ float3 Light; // = make_float3(0.0,1.2,3.75);   //Point Light-Source Position
	__device__ float4 spheres[2]; // = {make_float4(1.0,0.0,4.0,0.5), make_float4(-0.6,-1.0,4.5,0.5)};         //Sphere Center & Radius
	__device__ float2 planes[5]; // = {make_float2(0, 1.5),make_float2(1, -1.5),make_float2(0, -1.5),make_float2(1, 1.5),make_float2(2,5.0)}; //Plane Axis & Distance-to-Origin

	__device__ int numPhotons[2][5]; // = {{0,0},{0,0,0,0,0}};              //Photon Count for Each Scene Object
	__device__ float3 photons[2][5][5000][3]; // = new float[2][5][5000][3][3]; //Allocated Memory for Per-Object Photon Info



	// ----- Photon Mapping -----
	__device__ int nrPhotons = 1000;             //Number of Photons Emitted
	__device__ int nrBounces = 3;                //Number of Times Each Photon Bounces
	__device__ bool lightPhotons = true;      //Enable Photon Lighting?
	__device__ float sqRadius = 0.7;             //Photon Integration Area (Squared for Efficiency)
	__device__ float exposure = 50.0;            //Number of Photons Integrated at Brightest Pixel

	__device__ float gAnimTime = 0.0f;



//---------------------------------------------------------------------------------------
//Ray-Geometry Intersections  -----------------------------------------------------------
//---------------------------------------------------------------------------------------

__device__ void checkDistance(float lDist, int p, int i, float & gDist, int & gType, int & gIndex, bool & gIntersect){
  if (lDist < gDist && lDist > 0.0){ //Closest Intersection So Far in Forward Direction of Ray?
    gType = p; gIndex = i; gDist = lDist; gIntersect = true;} //Save Intersection in Global State
}

__device__ void raySphere(int idx, float3 r, float3 o,
						  float & gDist, int & gType, int & gIndex, bool & gIntersect) //Ray-Sphere Intersection: r=Ray Direction, o=Ray Origin
{ 
  float3 s = make_float3(spheres[idx].x, spheres[idx].x, spheres[idx].z) - o;  //s=Sphere Center Translated into Coordinate Frame of Ray Origin
  float radius = spheres[idx].w;    //radius=Sphere Radius
  
  //Intersection of Sphere and Line     =       Quadratic Function of Distance
  float A = dot(r,r);                       // Remember This From High School? :
  float B = -2.0 * dot(s,r);                //    A x^2 +     B x +               C  = 0
  float C = dot(s,s) - pow(radius,2.0f);          // (r'r)x^2 - (2s'r)x + (s's - radius^2) = 0
  float D = B*B - 4*A*C;                     // Precompute Discriminant
  
  if (D > 0.0){                              //Solution Exists only if sqrt(D) is Real (not Imaginary)
    float sign = (C < -0.00001) ? 1 : -1;    //Ray Originates Inside Sphere If C < 0
    float lDist = (-B + sign*sqrt(D))/(2*A); //Solve Quadratic Equation for Distance to Intersection
    checkDistance(lDist,0,idx,gDist,gType,gIndex,gIntersect);				 //Is This Closest Intersection So Far?
  }             
}


__device__ void rayPlane(int idx, float3 r, float3 o,
						 float & gDist, int & gType, int & gIndex, bool & gIntersect){ //Ray-Plane Intersection
  int axis = (int) planes[idx].x;            //Determine Orientation of Axis-Aligned Plane
  
  switch(axis) {
	case 0:
		if (r.x != 0.0){                        //Parallel Ray -> No Intersection
			float lDist = (planes[idx].y - o.x) / r.x; //Solve Linear Equation (rx = p-o)
			checkDistance(lDist,1,idx,gDist,gType,gIndex,gIntersect);
		}
		break;
	case 1:
		if (r.y != 0.0){                        //Parallel Ray -> No Intersection
			float lDist = (planes[idx].y - o.y) / r.y; //Solve Linear Equation (rx = p-o)
			checkDistance(lDist,1,idx,gDist,gType,gIndex,gIntersect);
		}
		break;
	case 2:
		if (r.z != 0.0){                        //Parallel Ray -> No Intersection
			float lDist = (planes[idx].y - o.z) / r.z; //Solve Linear Equation (rx = p-o)
			checkDistance(lDist,1,idx,gDist,gType,gIndex,gIntersect);
		}
		break;
  }
  
  
}

__device__ void rayObject(int type, int idx, float3 r, float3 o,
						  float & gDist, int & gType, int & gIndex, bool & gIntersect){
  if (type == 0) 
	  raySphere(idx,r,o,gDist,gType,gIndex,gIntersect); 
  else 
	  rayPlane(idx,r,o,gDist,gType,gIndex,gIntersect);
}



//---------------------------------------------------------------------------------------
// Lighting -----------------------------------------------------------------------------
//---------------------------------------------------------------------------------------

__device__ float lightDiffuse(float3 N, float3 P){  //Diffuse Lighting at Point P with Surface Normal N
  float3 L = normalize( (Light - P) ); //Light Vector (Point to Light)
  return dot(N,L);                        //Dot Product = cos (Light-to-Surface-Normal Angle)
}



__device__ float3 sphereNormal(int idx, float3 P){
  return normalize((P - make_float3(spheres[idx].x,spheres[idx].y, spheres[idx].z))); //Surface Normal (Center to Point)
}

__device__ float3 planeNormal(int idx, float3 P, float3 O){
  int axis = (int) planes[idx].x;

  float3 N = make_float3(0.0,0.0,0.0);
  switch(axis) {
	case 0:
	  N.x = O.x - planes[idx].y;      //Vector From Surface to Light
	  break;
	case 1:
	  N.y = O.y - planes[idx].y;      //Vector From Surface to Light
	  break;
	case 2:
	  N.z = O.z - planes[idx].y;      //Vector From Surface to Light
	  break;
  }
  return normalize(N);
}

__device__ float3 surfaceNormal(int type, int index, float3 P, float3 Inside){
  if (type == 0) {return sphereNormal(index,P);}
  else           {return planeNormal(index,P,Inside);}
}

__device__ float lightObject(int type, int idx, float3 P, float lightAmbient){
  float i = lightDiffuse( surfaceNormal(type, idx, P, Light) , P );
  return min(1.0, max(i, lightAmbient));   //Add in Ambient Light by Constraining Min Value
}



//---------------------------------------------------------------------------------------
// Raytracing ---------------------------------------------------------------------------
//---------------------------------------------------------------------------------------

__device__ void raytrace(float3 ray, float3 origin,
						 float & gDist, int & gType, int & gIndex, bool & gIntersect)
{
  gIntersect = false; //No Intersections Along This Ray Yet
  gDist = 999999.9;   //Maximum Distance to Any Object
  
  for (int t = 0; t < nrTypes; t++)
    for (int i = 0; i < nrObjects[t]; i++)
      rayObject(t,i,ray,origin,gDist,gType,gIndex,gIntersect);
}



__device__ bool gatedSqDist3(float3 a, float3 b, float sqradius,
							 float & gSqDist){ //Gated Squared Distance
  float c = a.x - b.x;          //Efficient When Determining if Thousands of Points
  float d = c*c;                  //Are Within a Radius of a Point (and Most Are Not!)
  if (d > sqradius) return false; //Gate 1 - If this dimension alone is larger than
  c = a.y - b.y;                //         the search radius, no need to continue
  d += c*c;
  if (d > sqradius) return false; //Gate 2
  c = a.z - b.z;
  d += c*c;
  if (d > sqradius) return false; //Gate 3
  gSqDist = d;      return true ; //Store Squared Distance Itself in Global State
}



__device__ float3 gatherPhotons(float3 p, int type, int id,
								float & gSqDist){
  float3 energy = make_float3(0.0,0.0,0.0);  
  float3 N = surfaceNormal(type, id, p, gOrigin);                   //Surface Normal at Current Point
  for (int i = 0; i < numPhotons[type][id]; i++){                    //Photons Which Hit Current Object
    if (gatedSqDist3(p,photons[type][id][i][0],sqRadius,gSqDist)){           //Is Photon Close to Point?
      float weight = max(0.0, -dot(N, photons[type][id][i][1] ));   //Single Photon Diffuse Lighting
      weight *= (1.0 - sqrt(gSqDist)) / exposure;                    //Weight by Photon-Point Distance
      energy = (energy + (photons[type][id][i][2] * weight)); //Add Photon's Energy to Total
   }
  } 
  return energy;
}



__device__ float3 filterColor(float3 rgbIn, float r, float g, float b){ //e.g. White Light Hits Red Wall
  float3 rgbOut = make_float3(r,g,b);
  rgbOut.x = min(rgbOut.x,rgbIn.x); //Absorb Some Wavelengths (R,G,B)
  rgbOut.y = min(rgbOut.y,rgbIn.y); //Absorb Some Wavelengths (R,G,B)
  rgbOut.z = min(rgbOut.z,rgbIn.z); //Absorb Some Wavelengths (R,G,B)
  return rgbOut;
}

__device__ float3 getColor(float3 rgbIn, int type, int index){ //Specifies Material Color of Each Object
  if      (type == 1 && index == 0) { return filterColor(rgbIn, 0.0, 1.0, 0.0);}
  else if (type == 1 && index == 2) { return filterColor(rgbIn, 1.0, 0.0, 0.0);}
  else                              { return filterColor(rgbIn, 1.0, 1.0, 1.0);}
}



__device__ float3 reflect3(float3 ray, float3 fromPoint, 
						   int & gType, int & gIndex, float3 & gPoint){                //Reflect Ray
  float3 N = surfaceNormal(gType, gIndex, gPoint, fromPoint);  //Surface Normal
  return normalize((ray - (N * (2 * dot(ray,N)))));     //Approximation to Reflection
}





__device__ float3 computePixelColor(float x, float y,
									float & gDist, int & gType, int & gIndex, bool & gIntersect,
									float3 & gPoint,
									float & gSqDist){
  float3 rgb = {0.0,0.0,0.0};
  float3 ray = make_float3(  x/szImg - 0.5 ,       //Convert Pixels to Image Plane Coordinates
                 -(y/szImg - 0.5), 1.0); //Focal Length = 1.0
  raytrace(ray, gOrigin,gDist,gType,gIndex,gIntersect);                //Raytrace!!! - Intersected Objects are Stored in Global State

  if (gIntersect){                       //Intersection                    
    gPoint = (ray * gDist);           //3D Point of Intersection
        
    if (gType == 0 && gIndex == 1){      //Mirror Surface on This Specific Object
      ray = reflect3(ray,gOrigin, gType,gIndex, gPoint);        //Reflect Ray Off the Surface
      raytrace(ray, gPoint, gDist,gType,gIndex,gIntersect);             //Follow the Reflected Ray
      if (gIntersect){ 
		  gPoint = ( (ray * gDist) + gPoint); 
	  }
	} //3D Point of Intersection
        
    if (lightPhotons){                   //Lighting via Photon Mapping
      rgb = gatherPhotons(gPoint,gType,gIndex, gSqDist);
	} else {                                //Lighting via Standard Illumination Model (Diffuse + Ambient)
      int tType = gType, tIndex = gIndex;//Remember Intersected Object
      float i = gAmbient;                //If in Shadow, Use Ambient Color of Original Object
      raytrace( (gPoint - Light) , Light, gDist,gType,gIndex,gIntersect);  //Raytrace from Light to Object
      if (tType == gType && tIndex == gIndex) //Ray from Light->Object Hits Object First?
        i = lightObject(gType, gIndex, gPoint, gAmbient); //Not In Shadow - Compute Lighting
      rgb.x=i; rgb.y=i; rgb.z=i;
      rgb = getColor(rgb,tType,tIndex);
	}
  }
  return rgb;
}



//---------------------------------------------------------------------------------------
//Photon Mapping ------------------------------------------------------------------------
//---------------------------------------------------------------------------------------


__device__ uint m_w = 6548;    /* must not be zero */
__device__ uint m_z = 316;    /* must not be zero */
 
__device__ uint get_random()
{
    m_z = 36969 * (m_z & 65535) + (m_z >> 16);
    m_w = 18000 * (m_w & 65535) + (m_w >> 16);
    return (m_z << 16) + m_w;  /* 32-bit result */
}

__device__ float randFloat(float max) {
	uint randInt = get_random();
	
	// rand [0..1]
	float randFloat = (float) ((int) randInt) / (float) 65535;
	
	// rand [0..2*max]
	randFloat = randFloat * 2*max;

	// rand [-max..max]
	randFloat = randFloat - max;

	return randFloat;
}

__device__ float3 rand3(float max) {
	return make_float3(randFloat(max),randFloat(max),randFloat(max));
}




__device__ void storePhoton(int type, int id, float3 location, float3 direction, float3 energy){
  photons[type][id][numPhotons[type][id]][0] = location;  //Location
  photons[type][id][numPhotons[type][id]][1] = direction; //Direction
  photons[type][id][numPhotons[type][id]][2] = energy;    //Attenuated Energy (Color)
  numPhotons[type][id]++;
}



__device__ void shadowPhoton(float3 ray,
							 float & gDist, int & gType, int & gIndex, bool & gIntersect,
							 float3 & gPoint){                               //Shadow Photons
  float3 shadow = make_float3(-0.25,-0.25,-0.25);
  float3 tPoint = gPoint; 
  int tType = gType, tIndex = gIndex;                         //Save State
  float3 bumpedPoint = (gPoint + (ray * 0.00001));      //Start Just Beyond Last Intersection
  raytrace(ray, bumpedPoint, gDist,gType,gIndex,gIntersect);                                 //Trace to Next Intersection (In Shadow)
  float3 shadowPoint = ( (ray * gDist) + bumpedPoint); //3D Point
  storePhoton(gType, gIndex, shadowPoint, ray, shadow);
  gPoint = tPoint; gType = tType; gIndex = tIndex;            //Restore State
}



__device__ void emitPhotons(float & gSqDist, float3 & gPoint,
							float & gDist, int & gType, int & gIndex, bool & gIntersect){
  // randomSeed(0);                               //Ensure Same Photons Each Time
  for (int t = 0; t < nrTypes; t++)            //Initialize Photon Count to Zero for Each Object
    for (int i = 0; i < nrObjects[t]; i++)
      numPhotons[t][i] = 0; 

  
  for (int i = 0; i < nrPhotons; i++){ //Draw 3x Photons For Usability
    int bounces = 1;
    float3 rgb = make_float3(1.0,1.0,1.0);               //Initial Photon Color is White
    float3 ray = normalize( rand3(1.0) );    //Randomize Direction of Photon Emission
    float3 prevPoint = Light;                 //Emit From Point Light Source
    
	
    //Spread Out Light Source, But Don't Allow Photons Outside Room/Inside Sphere
    int k = 0;
	while (prevPoint.y >= Light.y && k < 1000){ 
	// for(int k = 0; k < 100; k++){
		prevPoint = (Light + (normalize(rand3(1.0)) * 0.75));
		k++;
	}
	
	
    if (abs(prevPoint.x) > 1.5 || abs(prevPoint.y) > 1.2 || gatedSqDist3(prevPoint,make_float3(spheres[0].x,spheres[0].y,spheres[0].z),spheres[0].w*spheres[0].w, gSqDist)) 
		bounces = nrBounces+1;
	

    raytrace(ray, prevPoint, gDist,gType,gIndex,gIntersect);                          //Trace the Photon's Path
    
	// bounces = 1;
	while (gIntersect && bounces <= 3) { // && bounces <= nrBounces){        //Intersection With New Object
        gPoint = ( (ray * gDist) + prevPoint);   //3D Point of Intersection
        rgb =  (getColor(rgb,gType,gIndex) * 1.0/sqrt((float)bounces));
        storePhoton(gType, gIndex, gPoint, ray, rgb);  //Store Photon Info 
        // drawPhoton(rgb, gPoint);                       //Draw Photon
        shadowPhoton(ray, gDist,gType,gIndex,gIntersect, gPoint);                             //Shadow Photon
        ray = reflect3(ray,prevPoint, gType,gIndex, gPoint);                  //Bounce the Photon
        raytrace(ray, gPoint, gDist,gType,gIndex,gIntersect);                         //Trace It to Next Location
        prevPoint = gPoint;
        bounces++;
	}
	
  }
  
}



__global__ void photon_mapping_kernel(uchar4* pos, unsigned int width, unsigned int height, float animTime) {

  // ----- Kernel indices -----
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int x = index%width;
  unsigned int y = index/width;


	// ----- Raytracing Globals -----
	bool gIntersect = false;       //For Latest Raytracing Call... Was Anything Intersected by the Ray?
	int gType;                        //... Type of the Intersected Object (Sphere or Plane)
	int gIndex;                       //... Index of the Intersected Object (Which Sphere/Plane Was It?)
	float gSqDist, gDist = -1.0;      //... Distance from Ray Origin to Intersection
	float3 gPoint; // = {0.0, 0.0, 0.0}; //... Point At Which the Ray Intersected the Object



  gOrigin = make_float3(0.0,0.0,0.0);
  
  Light = make_float3(0.0,1.2,3.75);

  spheres[0] = make_float4(1.0,0.0,4.0,0.5);
  spheres[1] = make_float4(-0.6,-1.0,4.5*cos(gAnimTime),0.5);

  planes[0] = make_float2(0, 1.5);
  planes[1] = make_float2(1, -1.5);
  planes[2] = make_float2(0, -1.5);
  planes[3] = make_float2(1, 1.5);
  planes[4] = make_float2(2,5.0);

  /*
  for(int i = 0; i < 2; i++) {
	  for(int j = 0; j < 5; j++) {
		numPhotons[i][j] = 0;
	  }
  }
  */

  gPoint = make_float3(0.0,0.0,0.0);

  if(index < width*height) {
	
	float3 pixelColor = computePixelColor(x,y, gDist, gType, gIndex, gIntersect,
									gPoint, gSqDist);

	
    //float3 pixelColor = make_float3(0.0,255.0,255.0);

	unsigned char r = (unsigned char)(pixelColor.x*255.0);
    unsigned char g = (unsigned char)(pixelColor.y*255.0);
    unsigned char b = (unsigned char)(pixelColor.z*255.0);
    
    // Each thread writes one pixel location in the texture (textel)
    pos[index].w = 0;
    pos[index].x = r;
    pos[index].y = g;
    pos[index].z = b;
  }

}

__global__ void emit_photons_kernel(float animTime){
	gAnimTime = animTime;


	// ----- Raytracing Globals -----
	bool gIntersect = false;       //For Latest Raytracing Call... Was Anything Intersected by the Ray?
	int gType;                        //... Type of the Intersected Object (Sphere or Plane)
	int gIndex;                       //... Index of the Intersected Object (Which Sphere/Plane Was It?)
	float gSqDist, gDist = -1.0;      //... Distance from Ray Origin to Intersection
	float3 gPoint; // = {0.0, 0.0, 0.0}; //... Point At Which the Ray Intersected the Object

	
	emitPhotons(gSqDist, gPoint,
							gDist, gType, gIndex,gIntersect);
	
}



extern "C" void launch_emit_photons_kernel(uchar4* pos, unsigned int image_width, 
							  unsigned int image_height, float animTime) {


 
	emit_photons_kernel<<< 1, 1>>>(animTime);

	cudaThreadSynchronize();

	checkCUDAError("kernel failed!");
}


extern "C" void launch_photon_mapping_kernel(uchar4* pos, unsigned int image_width, 
							  unsigned int image_height, float animTime) {


    int nThreads=256;
	int totalThreads = image_height * image_width;
	int nBlocks = totalThreads/nThreads; 
	nBlocks += ((totalThreads%nThreads)>0)?1:0;

	// emit_photons_kernel<<< 1, 1>>>(animTime);

	photon_mapping_kernel<<< nBlocks, nThreads>>>(pos, image_width, image_height, animTime);

	cudaThreadSynchronize();

	checkCUDAError("kernel failed!");
}


