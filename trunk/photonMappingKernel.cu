//kernelPBO.cu (Rob Farber)

#include <stdio.h>
#include <cutil_math.h>
#include <cutil_inline.h>

#define nrPhotons 1000
#define NR_PARTICLES 1000
#define MAX_VOL_PHOTONS_PER_THREAD 100

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
	__device__ int nrObjects [] = {3,5};          //2 Spheres, 5 Planes
	__device__ float gAmbient = 0.1;             //Ambient Lighting
	__device__ float3 gOrigin; // = make_float3(0.0,0.0,0.0);  //World Origin for Convenient Re-Use Below (Constant)
	__device__ float3 Light = {0.0,1.4,3.5};   //Point Light-Source Position
	__device__ float4 spheres[] = {{1.0,-1.0,4.0,0.5}, {-0.6,-1.0,4.5,0.5}, {0.0,0.0,2.5,0.8}};         //Sphere Center & Radius
	__device__ float4 particles[NR_PARTICLES];
	__device__ float2 planes[] = {{0, 1.5},{1, -1.5},{0, -1.5},{1, 1.5},{2,5.0}}; //Plane Axis & Distance-to-Origin

	__device__ int numPhotons[2][5]; // = {{0,0},{0,0,0,0,0}};              //Photon Count for Each Scene Object
	__device__ float3 photons[2][5][5000][3]; // = new float[2][5][5000][3][3]; //Allocated Memory for Per-Object Photon Info

	__device__ int numVolumePhotons[nrPhotons];   // number of photons stored per thread
	__device__ float3 volumePhotons[nrPhotons][MAX_VOL_PHOTONS_PER_THREAD];  // volume "photon map"
	
	__device__ float3 randomNumbers[10000]; // sequence of random numbers to re-use

	// ----- Photon Mapping -----
	// __device__ int nrPhotons = 1000;             //Number of Photons Emitted
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
  float3 s = make_float3(spheres[idx].x, spheres[idx].y, spheres[idx].z) - o;  //s=Sphere Center Translated into Coordinate Frame of Ray Origin
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




__device__ void rayParticle(int idx, float3 r, float3 o,
						  float & gDist, int & gType, int & gIndex, bool & gIntersect) //Ray-Sphere Intersection: r=Ray Direction, o=Ray Origin
{ 
  float3 s = make_float3(particles[idx].x, particles[idx].y, particles[idx].z) - o;  //s=Sphere Center Translated into Coordinate Frame of Ray Origin
  float radius = particles[idx].w;    //radius=Sphere Radius
  
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


__device__ float3 sphereParticleNormal(int idx, float3 P){
  return normalize((P - make_float3(particles[idx].x,particles[idx].y, particles[idx].z))); //Surface Normal (Center to Point)
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



__device__ float3 particleNormal(int type, int index, float3 P, float3 Inside){
  return sphereParticleNormal(index,P);
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


__device__ void particleRayTrace(float3 ray, float3 origin,
						 float & gDist, int & gType, int & gIndex, bool & gIntersect)
{
	gIntersect = false; //No Intersections Along This Ray Yet
	gDist = 999999.9;   //Maximum Distance to Any Object
  
	for (int i = 0; i < NR_PARTICLES; i++)
	  rayParticle(i,ray,origin,gDist,gType,gIndex,gIntersect);
}


__device__ void raytrace(float3 ray, float3 origin,
						 float & gDist, int & gType, int & gIndex, bool & gIntersect, bool ignoreSmoke)
{
	gIntersect = false; //No Intersections Along This Ray Yet
	gDist = 999999.9;   //Maximum Distance to Any Object

	for (int t = 0; t < nrTypes; t++) {
		for (int i = 0; i < nrObjects[t]; i++) {
			if(i == 2 && t == 0 && ignoreSmoke) {
				continue;
			}
			rayObject(t,i,ray,origin,gDist,gType,gIndex,gIntersect);
		}
	}
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


__device__ float3 gatherVolumePhotons(float3 p, int type, int id, float & gSqDist){
	
	float3 energy = make_float3(0.0,0.0,0.0);  
	float3 N = surfaceNormal(type, id, p, gOrigin);                   //Surface Normal at Current Point
	
	for (int i = 0; i < nrPhotons; i++){
		for(int j = 0; j < numVolumePhotons[i]; j++) {
			//Photons Which Hit Current Object
			if (gatedSqDist3(p,volumePhotons[i][j],sqRadius,gSqDist)){           //Is Photon Close to Point?
				energy = energy + 0.1*make_float3(1.0,1.0,1.0);
			}
		}		
	} 
	
	return energy;							
}


__device__ float3 gatherPhotons(float3 p, int type, int id,
								float & gSqDist){
  float3 energy = make_float3(0.0,0.0,0.0);  
  float3 N = surfaceNormal(type, id, p, gOrigin);                   //Surface Normal at Current Point
  for (int i = 0; i < numPhotons[type][id]; i++){
	
  if(photons[type][id][i][1].x == 0.0 && photons[type][id][i][1].y == 0.0 && photons[type][id][i][1].z == 0.0){
	continue;
  }
	//Photons Which Hit Current Object
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

/*
__device__ float3 refract3(float3 ray, float3 fromPoint,
						  int & gType, int & gIndex, float3 & gPoint, float factor) {

	float n1 = 1.0;  // refraction index of first material (air)
	float n2 = 1.3;  // refraction index of second material

	float n = n1 / n2;	
	
	float3 normal = factor * surfaceNormal(gType, gIndex, gPoint, fromPoint);  //Surface Normal

	float cosI = dot(normal, ray);

	float sinT2 = n * n * (1.0 - cosI*cosI);

	if(sinT2 > 1.0) {
		return make_float3(0.0,0.0,0.0);
	}

	return n * ray - (n*cosI + sqrt(1.0 - sinT2)) * normal;


}
*/

__device__ float3 refract3(float3 ray, float3 fromPoint,
							int & gType, int & gIndex, float3 & gPoint, float factor){                //Refract Ray
	
	float3 refractedRay = make_float3(0.0,0.0,0.0);

	// if(gType == 0) {  // only first sphere refracts rays
		
		float3 normal = factor * surfaceNormal(gType, gIndex, gPoint, fromPoint);  //Surface Normal
		
		float n1 = 1.0;  // refraction index of first material (air)
		float n2 = 1.3;  // refraction index of second material

		float n = n1 / n2;
		
		if(factor == -1.0) {
			n = 1.0;
		}

		float cosI = - dot(normal, ray);
		float cosT2 = 1.0 - (n * n * (1.0 - cosI * cosI));

		if (cosT2 > 0.0)
		{
			return (n * ray) + (n * cosI - sqrt(cosT2)) * normal;
			
		}
		
		return make_float3(0.0,0.0,0.0);

	//}

	//return refractedRay;

}






__device__ float3 particleReflect3(float3 ray, float3 fromPoint, 
						   int & gType, int & gIndex, float3 & gPoint, float factor){                //Reflect Ray
  float3 N = factor * particleNormal(gType, gIndex, gPoint, fromPoint);  //Surface Normal
  return normalize((ray - (N * (2 * dot(ray,N)))));     //Approximation to Reflection
}




__device__ float3 reflect3(float3 ray, float3 fromPoint, 
						   int & gType, int & gIndex, float3 & gPoint, float factor){                //Reflect Ray
  float3 N = factor * surfaceNormal(gType, gIndex, gPoint, fromPoint);  //Surface Normal
  return normalize((ray - (N * (2 * dot(ray,N)))));     //Approximation to Reflection
}




__device__ void handleRefraction4(float3 & ray, float3 & gOrigin, int & gType, int & gIndex, 
								 float3 & gPoint, bool & gIntersect, float & gDist) {
	ray = refract3(ray,gPoint, gType,gIndex, gPoint, 1.0);        //Refract Ray Off the Surface
	gPoint = ( (ray * 0.00001) + gPoint);
	raytrace(ray, gPoint, gDist,gType,gIndex,gIntersect,true);             //Follow the Refracted Ray
	gPoint = ( (ray * gDist) + gPoint);

	if(gIntersect && gType == 0 && gIndex == 0) {
		
		ray = refract3(ray,gPoint, gType,gIndex, gPoint, -1.0);        //Refract Ray Off from inside the Surface
		gPoint = ( (ray * 0.00001) + gPoint);
		raytrace(ray, gPoint, gDist,gType,gIndex,gIntersect,true);             //Follow the Reflected Ray
		gPoint = ( (ray * gDist) + gPoint);

	}
}







__device__ void handleReflection4(float3 & ray, float3 & gOrigin, int & gType, int & gIndex, 
								 float3 & gPoint, bool & gIntersect, float & gDist) {
	ray = reflect3(ray,gOrigin, gType,gIndex, gPoint, 1.0);        //Reflect Ray Off the Surface
	raytrace(ray, gPoint, gDist,gType,gIndex,gIntersect,true);             //Follow the Reflected Ray
	if (gIntersect){ 
		gPoint = ( (ray * gDist) + gPoint);
		if(gType == 0 && gIndex == 0) {
			handleRefraction4(ray, gOrigin, gType, gIndex, gPoint, gIntersect, gDist);
		}
	}
}




__device__ void handleRefraction3(float3 & ray, float3 & gOrigin, int & gType, int & gIndex, 
								 float3 & gPoint, bool & gIntersect, float & gDist) {
	ray = refract3(ray,gPoint, gType,gIndex, gPoint, 1.0);        //Refract Ray Off the Surface
	gPoint = ( (ray * 0.00001) + gPoint);
	raytrace(ray, gPoint, gDist,gType,gIndex,gIntersect,true);             //Follow the Refracted Ray
	gPoint = ( (ray * gDist) + gPoint);

	if(gIntersect && gType == 0 && gIndex == 0) {
		
		ray = refract3(ray,gPoint, gType,gIndex, gPoint, -1.0);        //Refract Ray Off from inside the Surface
		gPoint = ( (ray * 0.00001) + gPoint);
		raytrace(ray, gPoint, gDist,gType,gIndex,gIntersect,true);             //Follow the Reflected Ray
		gPoint = ( (ray * gDist) + gPoint);

		if (gType == 0 && gIndex == 1){      //Mirror Surface on This Specific Object
			handleReflection4(ray, gOrigin, gType, gIndex, gPoint, gIntersect, gDist);
		}

	}
}





__device__ void handleReflection3(float3 & ray, float3 & gOrigin, int & gType, int & gIndex, 
								 float3 & gPoint, bool & gIntersect, float & gDist) {
	ray = reflect3(ray,gOrigin, gType,gIndex, gPoint, 1.0);        //Reflect Ray Off the Surface
	raytrace(ray, gPoint, gDist,gType,gIndex,gIntersect,true);             //Follow the Reflected Ray
	if (gIntersect){ 
		gPoint = ( (ray * gDist) + gPoint);
		if(gType == 0 && gIndex == 0) {
			handleRefraction3(ray, gOrigin, gType, gIndex, gPoint, gIntersect, gDist);
		}
	}
}






__device__ void handleRefraction2(float3 & ray, float3 & gOrigin, int & gType, int & gIndex, 
								 float3 & gPoint, bool & gIntersect, float & gDist) {
	ray = refract3(ray,gPoint, gType,gIndex, gPoint, 1.0);        //Refract Ray Off the Surface
	gPoint = ( (ray * 0.00001) + gPoint);
	raytrace(ray, gPoint, gDist,gType,gIndex,gIntersect,true);             //Follow the Refracted Ray
	gPoint = ( (ray * gDist) + gPoint);

	if(gIntersect && gType == 0 && gIndex == 0) {
		
		ray = refract3(ray,gPoint, gType,gIndex, gPoint, -1.0);        //Refract Ray Off from inside the Surface
		gPoint = ( (ray * 0.00001) + gPoint);
		raytrace(ray, gPoint, gDist,gType,gIndex,gIntersect,true);             //Follow the Reflected Ray
		gPoint = ( (ray * gDist) + gPoint);

		if (gType == 0 && gIndex == 1){      //Mirror Surface on This Specific Object
			handleReflection3(ray, gOrigin, gType, gIndex, gPoint, gIntersect, gDist);
		}

	}
}





__device__ void handleReflection2(float3 & ray, float3 & gOrigin, int & gType, int & gIndex, 
								 float3 & gPoint, bool & gIntersect, float & gDist) {
	ray = reflect3(ray,gOrigin, gType,gIndex, gPoint, 1.0);        //Reflect Ray Off the Surface
	raytrace(ray, gPoint, gDist,gType,gIndex,gIntersect,true);             //Follow the Reflected Ray
	if (gIntersect){ 
		gPoint = ( (ray * gDist) + gPoint);
		if(gType == 0 && gIndex == 0) {
			handleRefraction2(ray, gOrigin, gType, gIndex, gPoint, gIntersect, gDist);
		}
	}
}




__device__ void handleRefraction(float3 & ray, float3 & gOrigin, int & gType, int & gIndex, 
								 float3 & gPoint, bool & gIntersect, float & gDist) {
	ray = refract3(ray,gPoint, gType,gIndex, gPoint, 1.0);        //Refract Ray Off the Surface
	gPoint = ( (ray * 0.00001) + gPoint);
	raytrace(ray, gPoint, gDist,gType,gIndex,gIntersect,true);             //Follow the Refracted Ray
	gPoint = ( (ray * gDist) + gPoint);

	if(gIntersect && gType == 0 && gIndex == 0) {
		
		ray = refract3(ray,gPoint, gType,gIndex, gPoint, -1.0);        //Refract Ray Off from inside the Surface
		gPoint = ( (ray * 0.00001) + gPoint);
		raytrace(ray, gPoint, gDist,gType,gIndex,gIntersect,true);             //Follow the Reflected Ray
		gPoint = ( (ray * gDist) + gPoint);

		if (gType == 0 && gIndex == 1){      //Mirror Surface on This Specific Object
			handleReflection2(ray, gOrigin, gType, gIndex, gPoint, gIntersect, gDist);
		}

	}
}




__device__ void handleReflection(float3 & ray, float3 & gOrigin, int & gType, int & gIndex, 
								 float3 & gPoint, bool & gIntersect, float & gDist) {
	ray = reflect3(ray,gOrigin, gType,gIndex, gPoint, 1.0);        //Reflect Ray Off the Surface
	raytrace(ray, gPoint, gDist,gType,gIndex,gIntersect,true);             //Follow the Reflected Ray
	if (gIntersect){ 
		gPoint = ( (ray * gDist) + gPoint);
		if(gType == 0 && gIndex == 0) {
			handleRefraction(ray, gOrigin, gType, gIndex, gPoint, gIntersect, gDist);
		}
	}
}




__device__ float3 computePixelColor(float x, float y,
									float & gDist, int & gType, int & gIndex, bool & gIntersect,
									float3 & gPoint,
									float & gSqDist){
  float3 rgb = make_float3(0.0,0.0,0.0);
  float3 ray = make_float3(  x/szImg - 0.5 ,       //Convert Pixels to Image Plane Coordinates
                 -(y/szImg - 0.5), 1.0); //Focal Length = 1.0
  raytrace(ray, gOrigin,gDist,gType,gIndex,gIntersect,false);                //Raytrace!!! - Intersected Objects are Stored in Global State

  if (gIntersect){                       //Intersection                    
    gPoint = (ray * gDist);           //3D Point of Intersection
    
	if(gType == 0 && gIndex == 2) {
		// intersected smoke bounding sphere
		// ray trace to particle sphere
		
		if(lightPhotons) {
		
			// ray marching
			
			int numSteps = 1;
			
			for(int i = 0; i < numSteps; i++) {
				gPoint = gPoint + ray*0.001;
				rgb += gatherVolumePhotons(gPoint,gType,gIndex, gSqDist);
			}

			// get out of smoke bounding sphere
			raytrace(ray, gOrigin,gDist,gType,gIndex,gIntersect,true);
			gPoint = (ray * gDist);
		
		} else {
			particleRayTrace(ray, gPoint,gDist,gType,gIndex,gIntersect);
		
			if(gIntersect){
				rgb = make_float3(1.0,0.0,0.0);
				gIntersect = false;
			} else {
				// get out of smoke bounding sphere
				raytrace(ray, gOrigin,gDist,gType,gIndex,gIntersect,true);
				gPoint = (ray * gDist);
			}
			
		}
		
	} 
	
	
	if(gIntersect) {
		
		if (gType == 0 && gIndex == 1){      //Mirror Surface on This Specific Object
			handleReflection(ray, gOrigin, gType, gIndex, gPoint, gIntersect, gDist);
		} else if(gType == 0 && gIndex == 0) {
			handleRefraction(ray, gOrigin, gType, gIndex, gPoint, gIntersect, gDist);
		}
		
		//3D Point of Intersection
		if(gIntersect) {
			if (lightPhotons){                   //Lighting via Photon Mapping
			  rgb += gatherPhotons(gPoint,gType,gIndex, gSqDist);
			} else {                                //Lighting via Standard Illumination Model (Diffuse + Ambient)
			  int tType = gType, tIndex = gIndex;//Remember Intersected Object
			  float i = gAmbient;                //If in Shadow, Use Ambient Color of Original Object
			  raytrace( (gPoint - Light) , Light, gDist,gType,gIndex,gIntersect,true);  //Raytrace from Light to Object
			  if (tType == gType && tIndex == gIndex) //Ray from Light->Object Hits Object First?
				i = lightObject(gType, gIndex, gPoint, gAmbient); //Not In Shadow - Compute Lighting
			  rgb.x=i; rgb.y=i; rgb.z=i;
			  rgb = getColor(rgb,tType,tIndex);
			}
		}
		
	
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
	float rnd = (float) ((int) randInt) / (float) 65535;
	
	// rand [0..2*max]
	rnd = rnd * 2*max;

	// rand [-max..max]
	rnd = rnd - max;

	return rnd;
}

__device__ float3 rand3(float max) {
	return make_float3(randFloat(max),randFloat(max),randFloat(max));
}




__device__ void storePhoton(int type, int id, float3 location, float3 direction, float3 energy, int index){
  photons[type][id][index][0] = location;  //Location
  photons[type][id][index][1] = direction; //Direction
  photons[type][id][index][2] = energy;    //Attenuated Energy (Color)
  
  if(index + 1 > numPhotons[type][id])  
	numPhotons[type][id] = index + 1;	// max possible photon count is max thread index
}

__device__ void addPhoton(int type, int id, float3 location, float3 direction, float3 energy, int index){
  photons[type][id][index][0] = location;  //Location
  photons[type][id][index][1] = direction; //Direction
  photons[type][id][index][2] += energy;    //Attenuated Energy (Color)
  
  if(index + 1 > numPhotons[type][id])  
	numPhotons[type][id] = index + 1;	// max possible photon count is max thread index
}



__device__ void shadowPhoton(float3 ray,
							 float & gDist, int & gType, int & gIndex, bool & gIntersect,
							 float3 & gPoint, int index){                               //Shadow Photons
  float3 shadow = make_float3(-0.25,-0.25,-0.25);
  float3 tPoint = gPoint; 
  int tType = gType, tIndex = gIndex;                         //Save State
  float3 bumpedPoint = (gPoint + (ray * 0.00001));      //Start Just Beyond Last Intersection
  raytrace(ray, bumpedPoint, gDist,gType,gIndex,gIntersect,true);                                 //Trace to Next Intersection (In Shadow)
  float3 shadowPoint = ( (ray * gDist) + bumpedPoint); //3D Point
  storePhoton(gType, gIndex, shadowPoint, ray, shadow, index);
  gPoint = tPoint; gType = tType; gIndex = tIndex;            //Restore State
}


__device__ void storeVolumePhoton(float3 gPoint, int index) {

	volumePhotons[index][numVolumePhotons[index]] = gPoint;
	numVolumePhotons[index]++;
}



__device__ void emitPhotons(int index, float & gSqDist, float3 & gPoint,
							float & gDist, int & gType, int & gIndex, bool & gIntersect){
   
  
	
  
  // randomSeed(0);                               //TODO: Ensure Same Photons Each Time
  /*
  for (int t = 0; t < nrTypes; t++)            //Initialize Photon Count to Zero for Each Object
    for (int i = 0; i < nrObjects[t]; i++)
      numPhotons[t][i] = 0; 
  */
  
	if(index < nrPhotons) {
		// for (int i = 0; i < 1000; i++){ //Draw 3x Photons For Usability
		int bounces = 1;
		float3 rgb = make_float3(1.0,1.0,1.0);               //Initial Photon Color is White
		// float3 ray = normalize( rand3(1.0) );    //Randomize Direction of Photon Emission
		float3 ray = normalize( randomNumbers[index] );    //Randomize Direction of Photon Emission
		float3 prevPoint = Light;                 //Emit From Point Light Source


		//Spread Out Light Source, But Don't Allow Photons Outside Room/Inside Sphere
		int k = 0;
		while (prevPoint.y >= Light.y && k < 10000){ 
		// for(int k = 0; k < 100; k++){
			// prevPoint = (Light + (normalize(rand3(1.0)) * 0.75));
			prevPoint = (Light + (normalize(randomNumbers[index+k]) * 0.75));
			k++;
		}


		if (abs(prevPoint.x) > 1.5 || abs(prevPoint.y) > 1.2 || 
			gatedSqDist3(prevPoint,make_float3(spheres[0].x,spheres[0].y,spheres[0].z),spheres[0].w*spheres[0].w, gSqDist)) {
			bounces = nrBounces+1;
		}

		raytrace(ray, Light, gDist,gType,gIndex,gIntersect,false);                          //Trace the Photon's Path
		// raytrace(ray, prevPoint, gDist,gType,gIndex,gIntersect);
		
		bool caustics = false;

		numVolumePhotons[index] = 0;
		
		while (gIntersect && bounces <= nrBounces){        //Intersection With New Object
			gPoint = ( (ray * gDist) + prevPoint);   //3D Point of Intersection
			
			
			if(gType == 0 && gIndex == 2) {
				// hit smoke bounding sphere
				
				// intersected smoke bounding sphere
				// ray trace to particle sphere
				particleRayTrace(ray, gPoint,gDist,gType,gIndex,gIntersect);
				
				int particleBounces = 0;
				
				while(gIntersect && particleBounces < 10) {
					
					// store volume photon
					
					storeVolumePhoton(gPoint, index);
					
					ray = reflect3(ray,prevPoint, gType,gIndex, gPoint, 1.0);                  //Bounce the Photon
					particleRayTrace(ray, gPoint,gDist,gType,gIndex,gIntersect);                         //Trace It to Next Location
					
					if(gIntersect) {
						gPoint = ( (ray * gDist) + gPoint);
					}
					
					particleBounces++;
				}
				
				// get out of smoke bounding sphere
				raytrace(ray, gPoint,gDist,gType,gIndex,gIntersect,true);
				gPoint = (ray * gDist);
				
				
			} else {
			
				if(caustics) {
					rgb = 4.0*make_float3(1.0,1.0,1.0);
					storePhoton(gType, gIndex, gPoint, ray, rgb, index);  //Store Photon Info 
				} else {
					rgb =  2.0*(getColor(rgb,gType,gIndex) * 1.0/sqrt((float)bounces));
					storePhoton(gType, gIndex, gPoint, ray, rgb, index);  //Store Photon Info 
					shadowPhoton(ray, gDist,gType,gIndex,gIntersect, gPoint, index);                             //Shadow Photon
				}
				
				// drawPhoton(rgb, gPoint);                       //Draw Photon
				
				
				if (gType == 0 && gIndex == 1){      //Mirror Surface on This Specific Object
					handleReflection(ray, prevPoint, gType, gIndex, gPoint, gIntersect, gDist);
					caustics = false;
				} else if(gType == 0 && gIndex == 0) {
					handleRefraction(ray, prevPoint, gType, gIndex, gPoint, gIntersect, gDist);
					caustics = true;
				} else {
					ray = reflect3(ray,prevPoint, gType,gIndex, gPoint, 1.0);                  //Bounce the Photon
					raytrace(ray, gPoint, gDist,gType,gIndex,gIntersect,true);                         //Trace It to Next Location
					caustics = false;
				}
				
				prevPoint = gPoint;
				
				bounces++;
				
				
			}
			
		}
	}
}




__device__ void positionObjects(float animTime) {
	
	spheres[0].x = -1.0*cos(animTime+2.3);
	// spheres[0].y = -1.0*cos(animTime);
	spheres[0].z = sin(animTime+2.3)+3.5;
	/*
	spheres[0].y = -1.0;
	spheres[0].z = 4.0;
	spheres[0].w = 0.5;
	*/
	spheres[1].x = 0.0;
	spheres[1].y = -1.0; //*sin(0.5*animTime+5.0);
	spheres[1].z = 3.5;

	/*
	spheres[1].w = 0.5;
	//spheres[1] = make_float4(-0.6,-1.0,4.5,0.5);

	planes[0] = make_float2(0, 1.5);
	planes[1] = make_float2(1, -1.5);
	planes[2] = make_float2(0, -1.5);
	planes[3] = make_float2(1, 1.5);
	planes[4] = make_float2(2,5.0);
	*/
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
  

  /*
  for(int i = 0; i < 2; i++) {
	  for(int j = 0; j < 5; j++) {
		numPhotons[i][j] = 0;
	  }
  }
  */

  positionObjects(animTime);

  gPoint = make_float3(0.0,0.0,0.0);

  if(index < width*height) {
	
	float3 pixelColor = computePixelColor(x,y, gDist, gType, gIndex, gIntersect,
									gPoint, gSqDist);

	
    //float3 pixelColor = make_float3(0.0,255.0,255.0);

	unsigned char r = (unsigned char)(pixelColor.x*255.0 > 255.0 ? 255.0 : pixelColor.x*255.0);
	unsigned char g = (unsigned char)(pixelColor.y*255.0 > 255.0 ? 255.0 : pixelColor.y*255.0);
	unsigned char b = (unsigned char)(pixelColor.z*255.0 > 255.0 ? 255.0 : pixelColor.z*255.0);
    
    // Each thread writes one pixel location in the texture (textel)
    pos[index].w = 0;
    pos[index].x = r;
    pos[index].y = g;
    pos[index].z = b;
  }

}

__global__ void emit_photons_kernel(float animTime){
	gAnimTime = animTime;

	int index = blockIdx.x * blockDim.x + threadIdx.x;


	// ----- Raytracing Globals -----
	bool gIntersect = false;       //For Latest Raytracing Call... Was Anything Intersected by the Ray?
	int gType;                        //... Type of the Intersected Object (Sphere or Plane)
	int gIndex;                       //... Index of the Intersected Object (Which Sphere/Plane Was It?)
	float gSqDist, gDist = -1.0;      //... Distance from Ray Origin to Intersection
	float3 gPoint; // = {0.0, 0.0, 0.0}; //... Point At Which the Ray Intersected the Object

	positionObjects(animTime);	

	emitPhotons(index, gSqDist, gPoint, gDist, gType, gIndex,gIntersect);
	
}


__global__ void init_random_numbers_kernel() {
	/*
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if(index < nrPhotons) {
		randomNumbers[index] = rand3(1.0);
	}
	*/



	for(int i = 0; i < 2000; i++) {
		randomNumbers[i] = rand3(1.0);
	}
}



__global__ void init_photons_kernel(float animTime) {

	positionObjects(animTime);

	for (int t = 0; t < nrTypes; t++)            //Initialize Photon Count to Zero for Each Object
		for (int i = 0; i < nrObjects[t]; i++)
		 numPhotons[t][i] = 0; 
}


__global__ void init_particles_kernel() {
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	particles[index].x = 0.5*sin((float)index+index*cos((float)index)); // + spheres[2].x;
	particles[index].y = 0.5*cos((float)index+index*sin((float)index)); // + spheres[2].y;
	particles[index].z = 0.5*sin((float)index+index*cos((float)index))+2.5;// randFloat(spheres[2].w) + spheres[2].z;
	particles[index].w = 0.01; // very small radius

}


extern "C" void launch_emit_photons_kernel(uchar4* pos, unsigned int image_width, 
							  unsigned int image_height, float animTime) {

	
	init_photons_kernel<<< 1,1>>>(animTime);

	cudaThreadSynchronize();

	checkCUDAError("kernel failed!");

	int nThreads=100;
	int totalThreads = nrPhotons;
	int nBlocks = totalThreads/nThreads; 
	nBlocks += ((totalThreads%nThreads)>0)?1:0;
 
	emit_photons_kernel<<< nBlocks, nThreads>>>(animTime);

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


extern "C" void launch_init_random_numbers_kernel() {

	
	int nThreads=100;
	int totalThreads = NR_PARTICLES;
	int nBlocks = totalThreads/nThreads; 
	nBlocks += ((totalThreads%nThreads)>0)?1:0;
	

	init_random_numbers_kernel<<< 1, 1 >>>();
	cudaThreadSynchronize();
	checkCUDAError("kernel failed!");


	init_particles_kernel<<< nBlocks, nThreads >>>();
	cudaThreadSynchronize();
	checkCUDAError("kernel failed!");
}


