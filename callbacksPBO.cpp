//callbacksPBO.cpp (Rob Farber)

#include <GL/glew.h>
#include <cuda_runtime.h>
#include <cutil_inline.h>
#include <cutil_gl_inline.h>
#include <cuda_gl_interop.h>


#include <cutil_math.h>

// cuda types
#include <vector_types.h>

//#include <cutil_gl_error.h>
#include <rendercheck_gl.h>

// variables for keyboard control
int animFlag=1;
int emitFlag=1;
float animTime=0.0f;
float animInc=0.1f;

//external variables
extern GLuint pbo;
extern GLuint textureID;
extern unsigned int image_width;
extern unsigned int image_height;

// The user must create the following routines:
void runCuda(int numPhotons [][5],
				  float *photons[2][5][5000][3]);

void renderCuda(float3 pixelData[]);

void photonMappingCuda();
void simpleRunCuda();


////////////////////////////////////////////////////////////////////////////
// Photon Mapping Code

/*

// ----- Scene Description -----

int szImg = 512;                  //Image Size
int nrTypes = 2;                  //2 Object Types (Sphere = 0, Plane = 1)

int nrObjects[] = {2,5};          //2 Spheres, 5 Planes
float gAmbient = 0.1;             //Ambient Lighting

float3 gOrigin = make_float3(0.0,0.0,0.0);  //World Origin for Convenient Re-Use Below (Constant)
float3 Light = make_float3(0.0,1.2,3.75);   //Point Light-Source Position

float spheres[][4] = {{1.0,0.0,4.0,0.5},{-0.6,-1.0,4.5,0.5}};         //Sphere Center & Radius
float planes[][2]  = {{0, 1.5},{1, -1.5},{0, -1.5},{1, 1.5},{2,5.0}}; //Plane Axis & Distance-to-Origin

float3 pixelData[512*512];

// ----- Photon Mapping -----
int nrPhotons = 1000;             //Number of Photons Emitted

int nrBounces = 3;                //Number of Times Each Photon Bounces
bool lightPhotons = true;      //Enable Photon Lighting?

float sqRadius = 0.7;             //Photon Integration Area (Squared for Efficiency)
float exposure = 50.0;            //Number of Photons Integrated at Brightest Pixel

int numPhotons [2][5]; // = {{0,0},{0,0,0,0,0}};              //Photon Count for Each Scene Object
float3 photons[2][5][5000][3]; // = new float[2][5][5000][3][3]; //Allocated Memory for Per-Object Photon Info


// ----- Raytracing Globals -----
bool gIntersect = false;       //For Latest Raytracing Call... Was Anything Intersected by the Ray?

int gType;                        //... Type of the Intersected Object (Sphere or Plane)
int gIndex;                       //... Index of the Intersected Object (Which Sphere/Plane Was It?)

float gSqDist, gDist = -1.0;      //... Distance from Ray Origin to Intersection
float3 gPoint;
//{0.0f, 0.0f, 0.0f}; //... Point At Which the Ray Intersected the Object




//Vector Operations ---------------------------------------------------------------------
//---------------------------------------------------------------------------------------


float random(float min, float max) {
	int randInt = rand();
	
	// rand [0..1]
	float randFloat = (float) randInt / (float) RAND_MAX;

	// rand [0..max-min]
	randFloat *= (max-min);

	// rand [min..max]
	randFloat += min;

	return randFloat;

}

float3 rand3(float s){               //Random 3-Vector

  float3 rand = make_float3(random(-s,s),random(-s,s),random(-s,s));

  return rand;
}

bool gatedSqDist3(float3 & a, float3 & b, float sqradius){ //Gated Squared Distance

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





//---------------------------------------------------------------------------------------
//Ray-Geometry Intersections  -----------------------------------------------------------

//---------------------------------------------------------------------------------------

void checkDistance(float lDist, int p, int i){

  if (lDist < gDist && lDist > 0.0){ //Closest Intersection So Far in Forward Direction of Ray?
    gType = p; gIndex = i; gDist = lDist; gIntersect = true;} //Save Intersection in Global State

}


void raySphere(int idx, float3 & r, float3 & o) //Ray-Sphere Intersection: r=Ray Direction, o=Ray Origin

{ 
  float3 s = make_float3(spheres[idx][0],spheres[idx][1],spheres[idx][2]) - o;  //s=Sphere Center Translated into Coordinate Frame of Ray Origin
  float radius = spheres[idx][3];    //radius=Sphere Radius

  
  //Intersection of Sphere and Line     =       Quadratic Function of Distance
  float A = dot(r,r);                       // Remember This From High School? :

  float B = -2.0 * dot(s,r);                //    A x^2 +     B x +               C  = 0
  float C = dot(s,s) - pow(radius, 2.0f);          // (r'r)x^2 - (2s'r)x + (s's - radius^2) = 0

  float D = B*B - 4*A*C;                     // Precompute Discriminant
  
  if (D > 0.0){                              //Solution Exists only if sqrt(D) is Real (not Imaginary)

    float sign = (C < -0.00001) ? 1 : -1;    //Ray Originates Inside Sphere If C < 0
    float lDist = (-B + sign*sqrt(D))/(2*A); //Solve Quadratic Equation for Distance to Intersection

    checkDistance(lDist,0,idx);}             //Is This Closest Intersection So Far?
}

void rayPlane(int idx, float3 & r, float3 & o){ //Ray-Plane Intersection

  int axis = (int) planes[idx][0];            //Determine Orientation of Axis-Aligned Plane


  switch(axis) {
	case 0:
		if (r.x != 0.0){                        //Parallel Ray -> No Intersection
			float lDist = (planes[idx][1] - o.x) / r.x; //Solve Linear Equation (rx = p-o)
			checkDistance(lDist,1,idx);
		}
		break;
	case 1:
		if (r.x != 0.0){                        //Parallel Ray -> No Intersection
			float lDist = (planes[idx][1] - o.y) / r.y; //Solve Linear Equation (rx = p-o)
			checkDistance(lDist,1,idx);
		}
		break;

	case 2:
		if (r.x != 0.0){                        //Parallel Ray -> No Intersection
			float lDist = (planes[idx][1] - o.z) / r.z; //Solve Linear Equation (rx = p-o)
			checkDistance(lDist,1,idx);
		}
		break;

  }
    
}

void rayObject(int type, int idx, float3 & r, float3 & o){

  if (type == 0) raySphere(idx,r,o); else rayPlane(idx,r,o);
}

float min(float a, float b) {
	if(a <= b) {
		return a;
	} else {
		return b;
	}
}

float max(float a, float b) {
	if(a >= b) {
		return a;
	} else {
		return b;
	}
}


//---------------------------------------------------------------------------------------
// Lighting -----------------------------------------------------------------------------

//---------------------------------------------------------------------------------------

float lightDiffuse(float3 & N, float3 & P){  //Diffuse Lighting at Point P with Surface Normal N

  float3 L = normalize( (Light - P) ); //Light Vector (Point to Light)
  return dot(N,L);                        //Dot Product = cos (Light-to-Surface-Normal Angle)

}

float3 sphereNormal(int idx, float3 & P){
  return normalize((P - make_float3(spheres[idx][0],spheres[idx][1],spheres[idx][2]))); //Surface Normal (Center to Point)

}

float3 & planeNormal(int idx, float3 & P, float3 & O){

  int axis = (int) planes[idx][0];
  float3 N = make_float3(0.0f,0.0f,0.0f);
  
  switch(axis) {
	case 0:
		N.x = O.x - planes[idx][1];      //Vector From Surface to Light
		break;
	case 1:
		N.y = O.y - planes[idx][1];      //Vector From Surface to Light
		break;
	case 2:
		N.z = O.z - planes[idx][1];      //Vector From Surface to Light
		break;

  }
  
  return normalize(N);
}

float3 & surfaceNormal(int type, int index, float3 & P, float3 & Inside){

  if (type == 0) {return sphereNormal(index,P);}
  else           {return planeNormal(index,P,Inside);}

}

float lightObject(int type, int idx, float3 & P, float lightAmbient){

  float i = lightDiffuse( surfaceNormal(type, idx, P, Light) , P );
  return min(1.0, max(i, lightAmbient));   //Add in Ambient Light by Constraining Min Value

}

float3 & filterColor(float3 & rgbIn, float r, float g, float b){ //e.g. White Light Hits Red Wall

  float3 rgbOut = make_float3(r,g,b);

  rgbOut.x = min(rgbOut.x,rgbIn.x); //Absorb Some Wavelengths (R,G,B)
  rgbOut.y = min(rgbOut.y,rgbIn.y); //Absorb Some Wavelengths (R,G,B)
  rgbOut.z = min(rgbOut.z,rgbIn.z); //Absorb Some Wavelengths (R,G,B)

  return rgbOut;
}

float3 & getColor(float3 & rgbIn, int type, int index){ //Specifies Material Color of Each Object

  if      (type == 1 && index == 0) { return filterColor(rgbIn, 0.0, 1.0, 0.0);}
  else if (type == 1 && index == 2) { return filterColor(rgbIn, 1.0, 0.0, 0.0);}

  else                              { return filterColor(rgbIn, 1.0, 1.0, 1.0);}
}


//---------------------------------------------------------------------------------------
// Raytracing ---------------------------------------------------------------------------

//---------------------------------------------------------------------------------------

void raytrace(float3 & ray, float3 & origin)

{
  gIntersect = false; //No Intersections Along This Ray Yet
  gDist = 999999.9;   //Maximum Distance to Any Object

  
  for (int t = 0; t < nrTypes; t++)
    for (int i = 0; i < nrObjects[t]; i++)

      rayObject(t,i,ray,origin);
}


void storePhoton(int type, int id, float3 & location, float3 & direction, float3 & energy){

  photons[type][id][numPhotons[type][id]][0] = location;  //Location
  photons[type][id][numPhotons[type][id]][1] = direction; //Direction

  photons[type][id][numPhotons[type][id]][2] = energy;    //Attenuated Energy (Color)
  numPhotons[type][id]++;
}

void shadowPhoton(float3 & ray){                               //Shadow Photons

  float3 shadow = make_float3(-0.25,-0.25,-0.25);
  float3 tPoint = gPoint; 
  int tType = gType, tIndex = gIndex;                         //Save State

  float3 bumpedPoint = (gPoint + (ray * 0.00001));      //Start Just Beyond Last Intersection
  raytrace(ray, bumpedPoint);                                 //Trace to Next Intersection (In Shadow)

  float3 shadowPoint = ( (ray*gDist) + bumpedPoint); //3D Point
  storePhoton(gType, gIndex, shadowPoint, ray, shadow);

  gPoint = tPoint; gType = tType; gIndex = tIndex;            //Restore State
}


float3 reflect3(float3 & ray, float3 & fromPoint){                //Reflect Ray

  float3 N = surfaceNormal(gType, gIndex, gPoint, fromPoint);  //Surface Normal
  return normalize((ray - (N*(2 * dot(ray,N)))));     //Approximation to Reflection

}


void emitPhotons(){
	
	// randomSeed(0);                               //Ensure Same Photons Each Time, really needed?

  for (int t = 0; t < nrTypes; t++)            //Initialize Photon Count to Zero for Each Object
    for (int i = 0; i < nrObjects[t]; i++)
      numPhotons[t][i] = 0; 

  for (int i = 0; i < nrPhotons * 3.0; i++){ //Draw 3x Photons For Usability

    int bounces = 1;
    float3 rgb = make_float3(1.0,1.0,1.0); //Initial Photon Color is White

    float3 ray = normalize( rand3(1.0) );    //Randomize Direction of Photon Emission
    float3 prevPoint = Light;                 //Emit From Point Light Source

    
    //Spread Out Light Source, But Don't Allow Photons Outside Room/Inside Sphere
    while (prevPoint.y >= Light.y){ prevPoint = (Light + (normalize(rand3(1.0)) * 0.75));}

    if (abs(prevPoint.x) > 1.5 || abs(prevPoint.y) > 1.2 ||
        gatedSqDist3(prevPoint,make_float3(spheres[0][0],spheres[0][1],spheres[0][2]),spheres[0][3]*spheres[0][3])) bounces = nrBounces+1;

    
    raytrace(ray, prevPoint);                          //Trace the Photon's Path
    
    while (gIntersect && bounces <= nrBounces){        //Intersection With New Object

        gPoint = ( (ray*gDist) + prevPoint);   //3D Point of Intersection
        rgb = (getColor(rgb,gType,gIndex) * 1.0/sqrt((float)bounces));

        storePhoton(gType, gIndex, gPoint, ray, rgb);  //Store Photon Info 
        // drawPhoton(rgb, gPoint);                       //Draw Photon, skipped

        shadowPhoton(ray);                             //Shadow Photon
        ray = reflect(ray,prevPoint);                  //Bounce the Photon

        raytrace(ray, gPoint);                         //Trace It to Next Location
        prevPoint = gPoint;
        bounces++;}
  }
}


////////////////////////////////////////////////////////////////////////////



float3 & gatherPhotons(float3 & p, int type, int id){
  float3 energy = make_float3(0.0,0.0,0.0);  
  float3 & N = surfaceNormal(type, id, p, gOrigin);                   //Surface Normal at Current Point
  for (int i = 0; i < numPhotons[type][id]; i++){                    //Photons Which Hit Current Object
    if (gatedSqDist3(p,photons[type][id][i][0],sqRadius)){           //Is Photon Close to Point?
      float weight = max(0.0, -dot(N, photons[type][id][i][1] ));   //Single Photon Diffuse Lighting
      weight *= (1.0 - sqrt(gSqDist)) / exposure;                    //Weight by Photon-Point Distance
      energy = (energy + (photons[type][id][i][2] * weight)); //Add Photon's Energy to Total
   }} 
  return energy;
}


float3 & computePixelColor(float x, float y){
  float3 rgb = make_float3(0.0,0.0,0.0);
  float3 ray = make_float3(  x/szImg - 0.5 ,       //Convert Pixels to Image Plane Coordinates
                 -(y/szImg - 0.5), 1.0); //Focal Length = 1.0
  raytrace(ray, gOrigin);                //Raytrace!!! - Intersected Objects are Stored in Global State

  if (gIntersect){                       //Intersection                    
    gPoint = (ray*gDist);           //3D Point of Intersection
        
    if (gType == 0 && gIndex == 1){      //Mirror Surface on This Specific Object
      ray = reflect(ray,gOrigin);        //Reflect Ray Off the Surface
      raytrace(ray, gPoint);             //Follow the Reflected Ray
      if (gIntersect){ 
		  gPoint = ( (ray * gDist) + gPoint); //3D Point of Intersection
	  }
	}
	
	rgb = gatherPhotons(gPoint,gType,gIndex);
 
  }
  return rgb;
}



void computePixelData() {
	
	int minx = 100;
	int miny = 100;

	int maxx = minx + 50;
	int maxy = miny + 50;

	for(int y = miny; y < maxy; y++) {
		for(int x = minx; x < maxx; x++) {
			pixelData[y*image_width + x] = computePixelColor(x,y);
		}
	}
}

*/

void display()
{
  /*
  printf("emitting photons...");
  printf("\n");
  emitPhotons();
  printf("done emitting photons.");
  printf("\n");

  printf("computing pixel data...");
  printf("\n");
  computePixelData();
  printf("done computing pixel data.");
  printf("\n");
  */
  // run CUDA kernel
  // runCuda(numPhotons, photons);

  // renderCuda(pixelData);
	
	if(emitFlag)
		simpleRunCuda();

  photonMappingCuda();

  // Create a texture from the buffer
  glBindBuffer( GL_PIXEL_UNPACK_BUFFER, pbo);

  // bind texture from PBO
  glBindTexture(GL_TEXTURE_2D, textureID);


  // Note: glTexSubImage2D will perform a format conversion if the
  // buffer is a different format from the texture. We created the
  // texture with format GL_RGBA8. In glTexSubImage2D we specified
  // GL_BGRA and GL_UNSIGNED_INT. This is a fast-path combination

  // Note: NULL indicates the data resides in device memory
  glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, image_width, image_height, 
		  GL_RGBA, GL_UNSIGNED_BYTE, NULL);


  // Draw a single Quad with texture coordinates for each vertex.

  glBegin(GL_QUADS);
  glTexCoord2f(0.0f,1.0f); glVertex3f(0.0f,0.0f,0.0f);
  glTexCoord2f(0.0f,0.0f); glVertex3f(0.0f,1.0f,0.0f);
  glTexCoord2f(1.0f,0.0f); glVertex3f(1.0f,1.0f,0.0f);
  glTexCoord2f(1.0f,1.0f); glVertex3f(1.0f,0.0f,0.0f);
  glEnd();

  // Don't forget to swap the buffers!
  glutSwapBuffers();

  // if animFlag is true, then indicate the display needs to be redrawn
  if(animFlag) {
    glutPostRedisplay();
    animTime += animInc;
  }
}

//! Keyboard events handler for GLUT
void keyboard(unsigned char key, int x, int y)
{
  switch(key) {
  case(27) :
    exit(0);
    break;
  case 'a': // toggle animation
  case 'A':
    animFlag = (animFlag)?0:1;
    break;

  case 'e': // toggle animation
  case 'E':
    emitFlag = (emitFlag)?0:1;
    break;
  case '-': // decrease the time increment for the CUDA kernel
    animInc -= 0.01;
    break;
  case '+': // increase the time increment for the CUDA kernel
    animInc += 0.01;
    break;
  case 'r': // reset the time increment 
    animInc = 0.01;
    break;
  }

  // indicate the display must be redrawn
  glutPostRedisplay();
}

// No mouse event handlers defined
void mouse(int button, int state, int x, int y)
{
}

void motion(int x, int y)
{
}