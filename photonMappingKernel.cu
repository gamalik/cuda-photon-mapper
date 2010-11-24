//kernelPBO.cu (Rob Farber)

#include <stdio.h>
#include <cutil_math.h>
#include <cutil_inline.h>

#define MAX_SEARCH_RADIUS 3
#define MAX_SPLAT_RADIUS 3
#define ENERGY_WEIGHT 0.00015
#define SPLAT_ENERGY_WEIGHT 0.05
#define CAUSTICS_PHOTONS 100

#define NR_PHOTONS_X 64
#define NR_PHOTONS_Y 64
#define NR_PHOTONS_Z 64

#define INTERPOLATE true

#define WORLD_DIM_X 3.0
#define WORLD_DIM_Y 3.0
#define WORLD_DIM_Z 6.0

#define NR_PARTICLES 1000
#define MAX_VOL_PHOTONS_PER_THREAD 1000
#define PARTICLE_RADIUS 0.01

#define nrPhotons 100000
#define RANDOM_NUMS 100000

#define X_AXIS 0
#define Y_AXIS 1
#define Z_AXIS 2

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
	__device__ float3 Light = {0.0,1.4,3.5};   //Point Light-Source Position
	__device__ float4 spheres[] = {{1.0,-1.0,1.0,0.4}, {-0.6,-1.0,4.5,0.4}, {0.0,0.0,1.5,1.0}};         //Sphere Center & Radius
	
	__device__ float2 planes[] = {
		{X_AXIS, WORLD_DIM_X / 2.0},
		{Y_AXIS, - WORLD_DIM_Y / 2.0},
		{X_AXIS, - WORLD_DIM_X / 2.0},
		{Y_AXIS, WORLD_DIM_Y / 2.0},
		{Z_AXIS, WORLD_DIM_Z}
	}; //Plane Axis & Distance-to-Origin

	
	__device__ float3 photons[NR_PHOTONS_X][NR_PHOTONS_Y][NR_PHOTONS_Z];  // store direction and energy at each point


	
	__device__ float3 randomNumbers[nrPhotons]; // sequence of random numbers to re-use

	// ----- Photon Mapping -----
	__device__ int nrBounces = 5;                //Number of Times Each Photon Bounces
	__device__ bool lightPhotons = true;      //Enable Photon Lighting?
	__device__ bool interpolate = INTERPOLATE;
	__device__ float sqRadius = 1.0;             //Photon Integration Area (Squared for Efficiency)
	__device__ float causticsSqRadius = 0.025;
	__device__ float volumeSqRadius = 0.05;
	__device__ float exposure = 50.0;            //Number of Photons Integrated at Brightest Pixel
	__device__ float volumePhotonWeight = 0.005;
	__device__ float atenuationWeight = 0.7;
	__device__ float photonWeight = 0.1;
	__device__ float causticsWeight = 0.1;
	__device__ float causticsFactor = 10.0;
	__device__ float difuseFactor = 5.0;


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
						 float & gDist, int & gType, int & gIndex, bool & gIntersect, bool ignoreSmoke)
{
	gIntersect = false; //No Intersections Along This Ray Yet
	gDist = 999999.9;   //Maximum Distance to Any Object

	for (int t = 0; t < nrTypes; t++) {
		for (int i = 0; i < nrObjects[t]; i++) {
			/*
			if(i == 2 && t == 0 && ignoreSmoke) {
				continue;
			}
			*/
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


__device__ int3 getVoxelCoordinates(float3 p) {
	
	int3 voxelPoint = make_int3(((p.x + WORLD_DIM_X / 2.0) / WORLD_DIM_X) * NR_PHOTONS_X, 
								((p.y + WORLD_DIM_Y / 2.0) / WORLD_DIM_Y) * NR_PHOTONS_Y, 
								 (p.z / WORLD_DIM_Z) * NR_PHOTONS_Z);
	return voxelPoint;

}

__device__ float3 getWorldCoordinates(int3 p) {
	float3 worldPoint = make_float3( ((((float)p.x) / (float)NR_PHOTONS_X) * WORLD_DIM_X) - WORLD_DIM_X / 2.0,  
									 ((((float)p.y) / (float)NR_PHOTONS_Y) * WORLD_DIM_Y) - WORLD_DIM_Y / 2.0, 
									 ((((float)p.z) / (float)NR_PHOTONS_Z) * WORLD_DIM_Z));
	return worldPoint;
}


__device__ float sqDistancef(float3 a, float3 b) {
	return sqrt((a.x-b.x)*(a.x-b.x) + (a.y-b.y)*(a.y-b.y) + (a.z-b.z)*(a.z-b.z));
}

__device__ float sqDistance(int3 a, int3 b) {
	return sqrt((float)((a.x-b.x)*(a.x-b.x) + (a.y-b.y)*(a.y-b.y) + (a.z-b.z)*(a.z-b.z)));
}


__device__ float3 computeEnergy(float3 energy, float3 p, int3 worldPoint, int i, int j, int k) {
	// float weight = max(0.0, -dot(N, photons[i][j][k][0] ));   //Single Photon Diffuse Lighting
	// weight *= (1.0 - sqrt(gSqDist)) / exposure;                    //Weight by Photon-Point Distance
	
	/*
	float dist = 0.0001*sqDistance(worldPoint, make_int3(i,j,k));
	weight -= dist;
	*/

	float3 totEnergy = (energy + (photons[i][j][k] * ENERGY_WEIGHT)); //Add Photon's Energy to Total
	//totEnergy -= dist*make_float3(1.0,1.0,1.0);
	

	/*
	float3 roundedCenter = getWorldCoordinates(worldPoint);
	roundedCenter += make_float3(((float) WORLD_DIM_X) / (NR_PHOTONS_X * 2.0),
								 ((float) WORLD_DIM_Y) / (NR_PHOTONS_Y * 2.0),
								 ((float) WORLD_DIM_Z) / (NR_PHOTONS_Z * 2.0));

	float distance = 0.001*sqDistancef(p, roundedCenter);
	totEnergy -= (distance*make_float3(1.0,1.0,1.0));
	*/

	return totEnergy;
}



__device__ float3 integrate(float3 p, float3 e, int3 worldPoint, int type, int id, float & gSqDist){
	
	float3 energy = e;

	int minX = 0;
	if(worldPoint.x - MAX_SEARCH_RADIUS >= 0) 
		minX = worldPoint.x - MAX_SEARCH_RADIUS;

	int minY = 0;
	if(worldPoint.y - MAX_SEARCH_RADIUS >= 0) 
		minY = worldPoint.y - MAX_SEARCH_RADIUS;

	int minZ = 0;
	if(worldPoint.z - MAX_SEARCH_RADIUS >= 0) 
		minZ = worldPoint.z - MAX_SEARCH_RADIUS;

	int maxX = NR_PHOTONS_X;
	if(worldPoint.x + MAX_SEARCH_RADIUS <= NR_PHOTONS_X)
		maxX = worldPoint.x + MAX_SEARCH_RADIUS;

	int maxY = NR_PHOTONS_Y;
	if(worldPoint.y + MAX_SEARCH_RADIUS <= NR_PHOTONS_Y)
		maxY = worldPoint.y + MAX_SEARCH_RADIUS;
	
	int maxZ = NR_PHOTONS_Z;
	if(worldPoint.z + MAX_SEARCH_RADIUS <= NR_PHOTONS_Z)
		maxZ = worldPoint.z + MAX_SEARCH_RADIUS;


	if(type == 1) {
		// search along plane axis

		if(id == 0 || id == 2) {
			// search along x-axis
			int i = worldPoint.x;
			
			if(id == 0)
				i = NR_PHOTONS_X - 1;
			else
				i = 0;

			for(int j = minY; j < maxY; j++) {
				for(int k = minZ; k < maxZ; k++) {
					energy = computeEnergy(energy, p, worldPoint, i, j, k);
				}
			}

		} else if(id == 1 || id == 3) {
			// search along y-axis
			int j = worldPoint.y;

			if(id == 1)
				j = 0;
			else
				j = NR_PHOTONS_Y - 1;

			for(int i = minX; i < maxX; i++) {
				for(int k = minZ; k < maxZ; k++) {
					energy = computeEnergy(energy, p, worldPoint, i, j, k);
				}
			}
		} else if(id == 4) {
			int k = NR_PHOTONS_Z - 1;

			for(int i = minX; i < maxX; i++) {
				for(int j = minY; j < maxY; j++) {	
					energy = computeEnergy(energy, p, worldPoint, i, j, k);
				}
			}
		}

	} 
	

	return energy;							
}


__device__ float3 centerPoint(float3 p) {

	int3 worldPoint = getVoxelCoordinates(p);
	float3 roundedCenter = getWorldCoordinates(worldPoint);
	roundedCenter += make_float3(((float) WORLD_DIM_X) / (NR_PHOTONS_X * 2.0),
								 ((float) WORLD_DIM_Y) / (NR_PHOTONS_Y * 2.0),
								 ((float) WORLD_DIM_Z) / (NR_PHOTONS_Z * 2.0));
	return roundedCenter;
}



__device__ float getAlpha(float3 p, float3 p1, float3 p2, int axis) {

	float alfa = 0.5;

	switch(axis) {
		case X_AXIS:
			alfa = (p.x - p1.x) / (p2.x - p1.x);
			break;
		case Y_AXIS:
			alfa = (p.y - p1.y) / (p2.y - p1.y);
			break;
		default:
			alfa = (p.z - p1.z) / (p2.z - p1.z);
			break;
	}

	return alfa;
}


__device__ float3 interpolate3f(float3 p1, float3 p2, float alfa) {
	return (1.0 - alfa) * p1 + alfa * p2;
}

__device__ float getVoxelDim(int axis) {

	switch(axis) {
		case X_AXIS:
			return ((float)WORLD_DIM_X / (float)NR_PHOTONS_X);
		case Y_AXIS:
			return ((float)WORLD_DIM_Y / (float)NR_PHOTONS_Y);
		default:
			return ((float)WORLD_DIM_Z / (float)NR_PHOTONS_Z);
	}
	
}


__device__ float3 bilinearInterpolate(float3 p, float3 p1, float3 p2, float3 p3, float3 p4,
									  float3 energy, int type, int id, float & gSqDist, 
									  int firstAxis, int secondAxis) {
	float3 c;

	int3 worldPoint1 = getVoxelCoordinates(p1);
	float3 c1 = integrate(p1, energy, worldPoint1, type, id, gSqDist);
	
	int3 worldPoint2 = getVoxelCoordinates(p2);
	float3 c2 = integrate(p2, energy, worldPoint2, type, id, gSqDist);
	
	int3 worldPoint3 = getVoxelCoordinates(p3);
	float3 c3 = integrate(p3, energy, worldPoint3, type, id, gSqDist);
	
	int3 worldPoint4 = getVoxelCoordinates(p4);
	float3 c4 = integrate(p4, energy, worldPoint4, type, id, gSqDist);

	float alfa = getAlpha(p, p1, p2, firstAxis);
	float3 p12 = interpolate3f(p1, p2, alfa);
	float3 c12 = interpolate3f(c1, c2, alfa);

	alfa = getAlpha(p, p3, p4, firstAxis);
	float3 p34 = interpolate3f(p3, p4, alfa);
	float3 c34 = interpolate3f(c3, c4, alfa);

	alfa = getAlpha(p, p12, p34, secondAxis);
	c = interpolate3f(c12, c34, alfa);

	return c;
}



__device__ float3 interpolateEnergy(float3 p, float3 energy, int type, int id, float & gSqDist) {

	float3 c = energy;

	if(type == 1) {
		// interpolate along plane axis
		if(id == 0 || id == 2) {
			// interpolate along x-axis
			
			float3 p1 = centerPoint(p);
			float3 p2 = make_float3(0.0,0.0,0.0);
			float3 p3 = make_float3(0.0,0.0,0.0);
			float3 p4 = make_float3(0.0,0.0,0.0);

			if(p.x > p1.z) {
				p2.z = p1.z + getVoxelDim(Z_AXIS);
				p3.z = p1.z + getVoxelDim(Z_AXIS);
				p4.z = p1.z;
			} else {
				p2.z = p1.z - getVoxelDim(Z_AXIS);
				p3.z = p1.z - getVoxelDim(Z_AXIS);
				p4.z = p1.z;
			}

			if(p.y > p1.y) {
				p2.y = p1.y;
				p3.y = p1.y + getVoxelDim(Y_AXIS);
				p4.y = p1.y + getVoxelDim(Y_AXIS);
			} else {
				p2.y = p1.y;
				p3.y = p1.y - getVoxelDim(Y_AXIS);
				p4.y = p1.y - getVoxelDim(Y_AXIS);
			}

			c = bilinearInterpolate(p, p1, p2, p3, p4,
									energy, type, id, gSqDist, 
									Z_AXIS, Y_AXIS);


		} else if(id == 1 || id == 3) {
			// interpolate along y-axis
			
			float3 p1 = centerPoint(p);
			float3 p2 = make_float3(0.0,0.0,0.0);
			float3 p3 = make_float3(0.0,0.0,0.0);
			float3 p4 = make_float3(0.0,0.0,0.0);

			if(p.x > p1.x) {
				p2.x = p1.x + getVoxelDim(X_AXIS);
				p3.x = p1.x + getVoxelDim(X_AXIS);
				p4.x = p1.x;
			} else {
				p2.x = p1.x - getVoxelDim(X_AXIS);
				p3.x = p1.x - getVoxelDim(X_AXIS);
				p4.x = p1.x;
			}

			if(p.z > p1.z) {
				p2.z = p1.z;
				p3.z = p1.z + getVoxelDim(Z_AXIS);
				p4.z = p1.z + getVoxelDim(Z_AXIS);
			} else {
				p2.z = p1.z;
				p3.z = p1.z - getVoxelDim(Z_AXIS);
				p4.z = p1.z - getVoxelDim(Z_AXIS);
			}

			c = bilinearInterpolate(p, p1, p2, p3, p4,
									energy, type, id, gSqDist, 
									X_AXIS, Z_AXIS);

		} else if(id == 4) {
			// interpolate along z-axis
			
			float3 p1 = centerPoint(p);
			float3 p2 = make_float3(0.0,0.0,0.0);
			float3 p3 = make_float3(0.0,0.0,0.0);
			float3 p4 = make_float3(0.0,0.0,0.0);

			if(p.x > p1.x) {
				p2.x = p1.x + getVoxelDim(X_AXIS);
				p3.x = p1.x + getVoxelDim(X_AXIS);
				p4.x = p1.x;
			} else {
				p2.x = p1.x - getVoxelDim(X_AXIS);
				p3.x = p1.x - getVoxelDim(X_AXIS);
				p4.x = p1.x;
			}

			if(p.y > p1.y) {
				p2.y = p1.y;
				p3.y = p1.y + getVoxelDim(Y_AXIS);
				p4.y = p1.y + getVoxelDim(Y_AXIS);
			} else {
				p2.y = p1.y;
				p3.y = p1.y - getVoxelDim(Y_AXIS);
				p4.y = p1.y - getVoxelDim(Y_AXIS);
			}

			c = bilinearInterpolate(p, p1, p2, p3, p4,
									energy, type, id, gSqDist, 
									X_AXIS, Y_AXIS);

		}
	} 

	return c;

}


__device__ float3 gatherPhotons(float3 p, int type, int id, float & gSqDist){
	
	float3 energy = make_float3(0.0,0.0,0.0);  
	float3 N = surfaceNormal(type, id, p, gOrigin);                   //Surface Normal at Current Point
	
	if(interpolate) {
		float3 c = interpolateEnergy(p, energy, type, id, gSqDist);
		return c;
	} else {
		int3 worldPoint = getVoxelCoordinates(p);
		float3 c = integrate(p, energy, worldPoint, type, id, gSqDist);
		return c;
	}

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


__device__ float3 refract3(float3 ray, float3 fromPoint,
							int & gType, int & gIndex, float3 & gPoint, float factor){                //Refract Ray
	
	float3 refractedRay = make_float3(0.0,0.0,0.0);

	// if(gType == 0) {  // only first sphere refracts rays
		
		float3 normal = factor * surfaceNormal(gType, gIndex, gPoint, fromPoint);  //Surface Normal
		//float3 normal = surfaceNormal(gType, gIndex, gPoint, fromPoint);  //Surface Normal
		
		float n1 = 1.0;  // refraction index of first material (air)
		float n2 = 1.3;  // refraction index of second material


		float n = n1 / n2;
		
		if(factor == -1.0) {
			n = 1.0;
			//n = n2/n1;
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
								 float3 & gPoint, bool & gIntersect, float & gDist, bool ignoreSmokeSphere) {
	ray = refract3(ray,gPoint, gType,gIndex, gPoint, 1.0);        //Refract Ray Off the Surface
	gPoint = ( (ray * 0.00001) + gPoint);
	raytrace(ray, gPoint, gDist,gType,gIndex,gIntersect,ignoreSmokeSphere);             //Follow the Refracted Ray
	gPoint = ( (ray * gDist) + gPoint);

	if(gIntersect && gType == 0 && gIndex == 0) {
		
		ray = refract3(ray,gPoint, gType,gIndex, gPoint, -1.0);        //Refract Ray Off from inside the Surface
		gPoint = ( (ray * 0.00001) + gPoint);
		raytrace(ray, gPoint, gDist,gType,gIndex,gIntersect,ignoreSmokeSphere);             //Follow the Reflected Ray
		gPoint = ( (ray * gDist) + gPoint);

		if (gType == 0 && gIndex == 1){      //Mirror Surface on This Specific Object
			handleReflection2(ray, gOrigin, gType, gIndex, gPoint, gIntersect, gDist);
		}

	}
}




__device__ void handleReflection(float3 & ray, float3 & gOrigin, int & gType, int & gIndex, 
								 float3 & gPoint, bool & gIntersect, float & gDist, bool ignoreSmokeSphere) {
	ray = reflect3(ray,gOrigin, gType,gIndex, gPoint, 1.0);        //Reflect Ray Off the Surface
	raytrace(ray, gPoint, gDist,gType,gIndex,gIntersect,ignoreSmokeSphere);             //Follow the Reflected Ray
	if (gIntersect){ 
		gPoint = ( (ray * gDist) + gPoint);
		if(gType == 0 && gIndex == 0) {
			handleRefraction(ray, gOrigin, gType, gIndex, gPoint, gIntersect, gDist, ignoreSmokeSphere);
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
    
	if(gIntersect) {
		
		if (gType == 0 && gIndex == 1){      //Mirror Surface on This Specific Object
			handleReflection(ray, gOrigin, gType, gIndex, gPoint, gIntersect, gDist, true);
		} else if(gType == 0 && gIndex == 0) {
			handleRefraction(ray, gOrigin, gType, gIndex, gPoint, gIntersect, gDist, true);
		}
		
		//3D Point of Intersection
		if(gIntersect) {
			if (lightPhotons){                   //Lighting via Photon Mapping
			  // rgb += make_float3(0.5,0.5,0.5);
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


__device__ void storeNeighborPhoton(float3 energy, int3 worldPoint, int i, int j, int k) {
	
	float dist = 1.0;
	int3 neighbor = make_int3(i,j,k);

	if(	worldPoint.x != neighbor.x ||
		worldPoint.y != neighbor.y ||
		worldPoint.z != neighbor.z) {
		dist = sqDistance(worldPoint, neighbor);
		photons[i][j][k] += SPLAT_ENERGY_WEIGHT*energy/dist;
	}

}


__device__ void splatEnergy(int3 voxelPoint, float3 energy, int type, int id) {

	int minX = 0;
	if(voxelPoint.x - MAX_SPLAT_RADIUS >= 0) 
		minX = voxelPoint.x - MAX_SPLAT_RADIUS;

	int minY = 0;
	if(voxelPoint.y - MAX_SPLAT_RADIUS >= 0) 
		minY = voxelPoint.y - MAX_SPLAT_RADIUS;

	int minZ = 0;
	if(voxelPoint.z - MAX_SPLAT_RADIUS >= 0) 
		minZ = voxelPoint.z - MAX_SPLAT_RADIUS;

	int maxX = NR_PHOTONS_X;
	if(voxelPoint.x + MAX_SPLAT_RADIUS <= NR_PHOTONS_X)
		maxX = voxelPoint.x + MAX_SPLAT_RADIUS;

	int maxY = NR_PHOTONS_Y;
	if(voxelPoint.y + MAX_SPLAT_RADIUS <= NR_PHOTONS_Y)
		maxY = voxelPoint.y + MAX_SPLAT_RADIUS;
	
	int maxZ = NR_PHOTONS_Z;
	if(voxelPoint.z + MAX_SPLAT_RADIUS <= NR_PHOTONS_Z)
		maxZ = voxelPoint.z + MAX_SPLAT_RADIUS;

	if(type == 1) {
		// search along plane axis

		if(id == 0 || id == 2) {
			// search along x-axis
			int i = voxelPoint.x;
			
			if(id == 0)
				i = NR_PHOTONS_X - 1;
			else
				i = 0;

			for(int j = minY; j < maxY; j++) {
				for(int k = minZ; k < maxZ; k++) {
					storeNeighborPhoton(energy, voxelPoint, i, j, k);
				}
			}

		} else if(id == 1 || id == 3) {
			// search along y-axis
			int j = voxelPoint.y;

			if(id == 1)
				j = 0;
			else
				j = NR_PHOTONS_Y - 1;

			for(int i = minX; i < maxX; i++) {
				for(int k = minZ; k < maxZ; k++) {
					storeNeighborPhoton(energy, voxelPoint, i, j, k);
				}
			}
		} else if(id == 4) {
			int k = NR_PHOTONS_Z - 1;

			for(int i = minX; i < maxX; i++) {
				for(int j = minY; j < maxY; j++) {	
					storeNeighborPhoton(energy, voxelPoint, i, j, k);
				}
			}
		}

	} 
	
}

__device__ void storePhoton(int type, int id, float3 location, float3 direction, float3 energy, int index){
  
  int3 voxelPoint = getVoxelCoordinates(location);

  voxelPoint.x = voxelPoint.x < NR_PHOTONS_X ? voxelPoint.x : NR_PHOTONS_X - 1;
  voxelPoint.x = voxelPoint.x < 0 ? 0 : voxelPoint.x;
  voxelPoint.y = voxelPoint.y < NR_PHOTONS_Y ? voxelPoint.y : NR_PHOTONS_Y - 1;
  voxelPoint.y = voxelPoint.y < 0 ? 0 : voxelPoint.y;
  voxelPoint.z = voxelPoint.z < NR_PHOTONS_Z ? voxelPoint.z : NR_PHOTONS_Z - 1;
  voxelPoint.z = voxelPoint.z < 0 ? 0 : voxelPoint.z;

  //photons[voxelPoint.x][voxelPoint.y][voxelPoint.z][0] = direction; //Direction
  photons[voxelPoint.x][voxelPoint.y][voxelPoint.z] += energy; // (energy + photonWeight*(photons[voxelPoint.x][voxelPoint.y][voxelPoint.z][1])); //Add Photon's Energy to Total

  splatEnergy(voxelPoint, energy, type, id);

  __threadfence();  // make sure every thread sees the change

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
		
		if(index < CAUSTICS_PHOTONS) {
			// aim first 2000 photons in the direction of crystal sphere
			ray = normalize( make_float3(spheres[0].x, spheres[0].y, spheres[0].z) - Light );
			ray += 0.01*normalize( randomNumbers[index] );
		}
		
		float3 prevPoint = Light;                 //Emit From Point Light Source

		

		raytrace(ray, Light, gDist,gType,gIndex,gIntersect,true);                          //Trace the Photon's Path
		//raytrace(ray, prevPoint, gDist,gType,gIndex,gIntersect,false);
		
		bool caustics = false;
		bool computeNewPoint = true;

		// numVolumePhotons = 0;
		
		while (gIntersect && bounces <= nrBounces){        //Intersection With New Object
			
			if(computeNewPoint)
				gPoint = ( (ray * gDist) + prevPoint);   //3D Point of Intersection
						
				if(caustics) {
					rgb = causticsFactor*make_float3(1.0,1.0,1.0);
					storePhoton(gType, gIndex, gPoint, ray, rgb, index);  //Store Photon Info 
				} else {
					rgb =  difuseFactor*(getColor(rgb,gType,gIndex) * 1.0/sqrt((float)bounces));
					storePhoton(gType, gIndex, gPoint, ray, rgb, index);  //Store Photon Info 
					shadowPhoton(ray, gDist,gType,gIndex,gIntersect, gPoint, index);                             //Shadow Photon
				}
				
				// drawPhoton(rgb, gPoint);                       //Draw Photon
				
				prevPoint = gPoint;

				
				if (gType == 0 && gIndex == 1){      //Mirror Surface on This Specific Object
					handleReflection(ray, prevPoint, gType, gIndex, gPoint, gIntersect, gDist, false);
					caustics = false;
					computeNewPoint = false;
				} else if(gType == 0 && gIndex == 0) {
					handleRefraction(ray, prevPoint, gType, gIndex, gPoint, gIntersect, gDist, false);
					caustics = true;
					computeNewPoint = false;
				} else {
					ray = reflect3(ray,prevPoint, gType,gIndex, gPoint, 1.0);                  //Bounce the Photon
					raytrace(ray, gPoint, gDist,gType,gIndex,gIntersect,false);                         //Trace It to Next Location
					caustics = false;
					computeNewPoint = true;
				}
				
				
				
				bounces++;
				
			
		}
	}
}




__device__ void positionObjects(float animTime) {
	
	spheres[0].x = 1.0*cos(animTime);
	// spheres[0].y = -1.0*cos(animTime);
	spheres[0].z = sin(animTime)+3.5;
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

	if(index < RANDOM_NUMS) {
		randomNumbers[index] = rand3(1.0);
	}
	*/


	
	for(int i = 0; i < nrPhotons; i++) {
		randomNumbers[i] = rand3(1.0);
	}
}




__global__ void init_photons_kernel(float animTime) {

	int i = blockIdx.x;
	int j = blockIdx.y;
	int k = threadIdx.x;

	
	if(	i >= 0 && i < NR_PHOTONS_X &&
		j >= 0 && j < NR_PHOTONS_Y &&
		k >= 0 && k < NR_PHOTONS_Z) {
				photons[i][j][k].x = 0.0;
				photons[i][j][k].y = 0.0;
				photons[i][j][k].z = 0.0;
	}

	positionObjects(animTime);

	
}

extern "C" void launch_emit_photons_kernel(uchar4* pos, unsigned int image_width, 
							  unsigned int image_height, float animTime) {

	
	int threadsPerBlock = NR_PHOTONS_Z;
	dim3 numBlocks(NR_PHOTONS_X, NR_PHOTONS_Y);
	
	init_photons_kernel<<< numBlocks, threadsPerBlock>>>(animTime);

	cudaThreadSynchronize();

	checkCUDAError("init_photons_kernel failed!");

	int nThreads=256;
	int totalThreads = nrPhotons;
	int nBlocks = totalThreads/nThreads; 
	nBlocks += ((totalThreads%nThreads)>0)?1:0;
 
	emit_photons_kernel<<< nBlocks, nThreads>>>(animTime);

	cudaThreadSynchronize();

	checkCUDAError("emit_photons_kernel failed!");
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

	checkCUDAError("photon_mapping_kernel failed!");
}


extern "C" void launch_init_random_numbers_kernel() {

	
	int nThreads=1;
	int totalThreads = 1;
	int nBlocks = totalThreads/nThreads; 
	nBlocks += ((totalThreads%nThreads)>0)?1:0;
	

	init_random_numbers_kernel<<< nBlocks, nThreads >>>();
	cudaThreadSynchronize();
	checkCUDAError("init_random_numbers_kernel failed!");

}


