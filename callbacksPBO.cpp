//callbacksPBO.cpp (Rob Farber)

#include <GL/glew.h>
#include <cuda_runtime.h>
#include <cutil_inline.h>
#include <cutil_gl_inline.h>
#include <cuda_gl_interop.h>
//#include <cutil_gl_error.h>
#include <rendercheck_gl.h>

// variables for keyboard control
int animFlag=1;
float animTime=0.0f;
float animInc=0.1f;

//external variables
extern GLuint pbo;
extern GLuint textureID;
extern unsigned int image_width;
extern unsigned int image_height;

// The user must create the following routines:
void runCuda();



////////////////////////////////////////////////////////////////////////////
// Photon Mapping Code



// ----- Scene Description -----

int szImg = 512;                  //Image Size
int nrTypes = 2;                  //2 Object Types (Sphere = 0, Plane = 1)

int nrObjects[] = {2,5};          //2 Spheres, 5 Planes
float gAmbient = 0.1;             //Ambient Lighting

float gOrigin[] = {0.0,0.0,0.0};  //World Origin for Convenient Re-Use Below (Constant)
float Light[] = {0.0,1.2,3.75};   //Point Light-Source Position

float spheres[][4] = {{1.0,0.0,4.0,0.5},{-0.6,-1.0,4.5,0.5}};         //Sphere Center & Radius
float planes[][2]  = {{0, 1.5},{1, -1.5},{0, -1.5},{1, 1.5},{2,5.0}}; //Plane Axis & Distance-to-Origin


// ----- Photon Mapping -----
int nrPhotons = 1000;             //Number of Photons Emitted

int nrBounces = 3;                //Number of Times Each Photon Bounces
boolean lightPhotons = true;      //Enable Photon Lighting?

float sqRadius = 0.7;             //Photon Integration Area (Squared for Efficiency)
float exposure = 50.0;            //Number of Photons Integrated at Brightest Pixel

int numPhotons [][5] = {{0,0},{0,0,0,0,0}};              //Photon Count for Each Scene Object
float *photons[2][5][5000][3]; // = new float[2][5][5000][3][3]; //Allocated Memory for Per-Object Photon Info


// ----- Raytracing Globals -----
boolean gIntersect = false;       //For Latest Raytracing Call... Was Anything Intersected by the Ray?

int gType;                        //... Type of the Intersected Object (Sphere or Plane)
int gIndex;                       //... Index of the Intersected Object (Which Sphere/Plane Was It?)

float gSqDist, gDist = -1.0;      //... Distance from Ray Origin to Intersection
float * gPoint = new float[3];
//{0.0f, 0.0f, 0.0f}; //... Point At Which the Ray Intersected the Object




//Vector Operations ---------------------------------------------------------------------
//---------------------------------------------------------------------------------------

float* mul3c(float a[], float c){    //Multiply 3-Vector with Scalar

  float result[] = {c*a[0], c*a[1], c*a[2]};
  return result;
}

float dot3(float a[], float b[]){     //Dot Product 3-Vectors

  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

float* normalize3(float v[]){        //Normalize 3-Vector
  float L = sqrt(dot3(v,v));

  return mul3c(v, 1.0/L);
}

float* sub3(float a[], float b[]){   //Subtract 3-Vectors

  float result[] = {a[0] - b[0], a[1] - b[1], a[2] - b[2]};
  return result;
}

float* add3(float a[], float b[]){   //Add 3-Vectors

  float result[] = {a[0] + b[0], a[1] + b[1], a[2] + b[2]};
  return result;
}

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

float* rand3(float s){               //Random 3-Vector

  float rand[] = {random(-s,s),random(-s,s),random(-s,s)};

  return rand;
}

bool gatedSqDist3(float a[], float b[], float sqradius){ //Gated Squared Distance

  float c = a[0] - b[0];          //Efficient When Determining if Thousands of Points
  float d = c*c;                  //Are Within a Radius of a Point (and Most Are Not!)

  if (d > sqradius) return false; //Gate 1 - If this dimension alone is larger than

  c = a[1] - b[1];                //         the search radius, no need to continue
  d += c*c;
  if (d > sqradius) return false; //Gate 2

  c = a[2] - b[2];
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


void raySphere(int idx, float r[], float o[]) //Ray-Sphere Intersection: r=Ray Direction, o=Ray Origin

{ 
  float * s = sub3(spheres[idx],o);  //s=Sphere Center Translated into Coordinate Frame of Ray Origin
  float radius = spheres[idx][3];    //radius=Sphere Radius

  
  //Intersection of Sphere and Line     =       Quadratic Function of Distance
  float A = dot3(r,r);                       // Remember This From High School? :

  float B = -2.0 * dot3(s,r);                //    A x^2 +     B x +               C  = 0
  float C = dot3(s,s) - pow(radius, 2.0f);          // (r'r)x^2 - (2s'r)x + (s's - radius^2) = 0

  float D = B*B - 4*A*C;                     // Precompute Discriminant
  
  if (D > 0.0){                              //Solution Exists only if sqrt(D) is Real (not Imaginary)

    float sign = (C < -0.00001) ? 1 : -1;    //Ray Originates Inside Sphere If C < 0
    float lDist = (-B + sign*sqrt(D))/(2*A); //Solve Quadratic Equation for Distance to Intersection

    checkDistance(lDist,0,idx);}             //Is This Closest Intersection So Far?
}

void rayPlane(int idx, float r[], float o[]){ //Ray-Plane Intersection

  int axis = (int) planes[idx][0];            //Determine Orientation of Axis-Aligned Plane

  if (r[axis] != 0.0){                        //Parallel Ray -> No Intersection
    float lDist = (planes[idx][1] - o[axis]) / r[axis]; //Solve Linear Equation (rx = p-o)

    checkDistance(lDist,1,idx);}
}

void rayObject(int type, int idx, float r[], float o[]){

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

float lightDiffuse(float N[], float P[]){  //Diffuse Lighting at Point P with Surface Normal N

  float * L = normalize3( sub3(Light,P) ); //Light Vector (Point to Light)
  return dot3(N,L);                        //Dot Product = cos (Light-to-Surface-Normal Angle)

}

float* sphereNormal(int idx, float* P){
  return normalize3(sub3(P,spheres[idx])); //Surface Normal (Center to Point)

}

float* planeNormal(int idx, float* P, float* O){

  int axis = (int) planes[idx][0];
  float N[] = {0.0f,0.0f,0.0f};
  N[axis] = O[axis] - planes[idx][1];      //Vector From Surface to Light

  return normalize3(N);
}

float* surfaceNormal(int type, int index, float* P, float* Inside){

  if (type == 0) {return sphereNormal(index,P);}
  else           {return planeNormal(index,P,Inside);}

}

float lightObject(int type, int idx, float* P, float lightAmbient){

  float i = lightDiffuse( surfaceNormal(type, idx, P, Light) , P );
  return min(1.0, max(i, lightAmbient));   //Add in Ambient Light by Constraining Min Value

}

float* filterColor(float* rgbIn, float r, float g, float b){ //e.g. White Light Hits Red Wall

  float rgbOut[] = {r,g,b};
  for (int c=0; c<3; c++) rgbOut[c] = min(rgbOut[c],rgbIn[c]); //Absorb Some Wavelengths (R,G,B)

  return rgbOut;
}

float* getColor(float rgbIn[], int type, int index){ //Specifies Material Color of Each Object

  if      (type == 1 && index == 0) { return filterColor(rgbIn, 0.0, 1.0, 0.0);}
  else if (type == 1 && index == 2) { return filterColor(rgbIn, 1.0, 0.0, 0.0);}

  else                              { return filterColor(rgbIn, 1.0, 1.0, 1.0);}
}


//---------------------------------------------------------------------------------------
// Raytracing ---------------------------------------------------------------------------

//---------------------------------------------------------------------------------------

void raytrace(float* ray, float* origin)

{
  gIntersect = false; //No Intersections Along This Ray Yet
  gDist = 999999.9;   //Maximum Distance to Any Object

  
  for (int t = 0; t < nrTypes; t++)
    for (int i = 0; i < nrObjects[t]; i++)

      rayObject(t,i,ray,origin);
}


void storePhoton(int type, int id, float* location, float* direction, float* energy){

  photons[type][id][numPhotons[type][id]][0] = location;  //Location
  photons[type][id][numPhotons[type][id]][1] = direction; //Direction

  photons[type][id][numPhotons[type][id]][2] = energy;    //Attenuated Energy (Color)
  numPhotons[type][id]++;
}

void shadowPhoton(float* ray){                               //Shadow Photons

  float shadow[] = {-0.25,-0.25,-0.25};
  float* tPoint = gPoint; 
  int tType = gType, tIndex = gIndex;                         //Save State

  float *bumpedPoint = add3(gPoint,mul3c(ray,0.00001));      //Start Just Beyond Last Intersection
  raytrace(ray, bumpedPoint);                                 //Trace to Next Intersection (In Shadow)

  float* shadowPoint = add3( mul3c(ray,gDist), bumpedPoint); //3D Point
  storePhoton(gType, gIndex, shadowPoint, ray, shadow);

  gPoint = tPoint; gType = tType; gIndex = tIndex;            //Restore State
}


float* reflect(float* ray, float* fromPoint){                //Reflect Ray

  float* N = surfaceNormal(gType, gIndex, gPoint, fromPoint);  //Surface Normal
  return normalize3(sub3(ray, mul3c(N,(2 * dot3(ray,N)))));     //Approximation to Reflection

}


void emitPhotons(){
	
	// randomSeed(0);                               //Ensure Same Photons Each Time, really needed?

  for (int t = 0; t < nrTypes; t++)            //Initialize Photon Count to Zero for Each Object
    for (int i = 0; i < nrObjects[t]; i++)
      numPhotons[t][i] = 0; 

  for (int i = 0; i < nrPhotons * 3.0; i++){ //Draw 3x Photons For Usability

    int bounces = 1;
    float * rgb = new float[3];
	
	rgb[0]=1.0;
	rgb[1]=1.0;
	rgb[2]=1.0;

	//{1.0,1.0,1.0};               //Initial Photon Color is White

    float * ray = normalize3( rand3(1.0) );    //Randomize Direction of Photon Emission
    float * prevPoint = Light;                 //Emit From Point Light Source

    
    //Spread Out Light Source, But Don't Allow Photons Outside Room/Inside Sphere
    while (prevPoint[1] >= Light[1]){ prevPoint = add3(Light, mul3c(normalize3(rand3(1.0)), 0.75));}

    if (abs(prevPoint[0]) > 1.5 || abs(prevPoint[1]) > 1.2 ||
        gatedSqDist3(prevPoint,spheres[0],spheres[0][3]*spheres[0][3])) bounces = nrBounces+1;

    
    raytrace(ray, prevPoint);                          //Trace the Photon's Path
    
    while (gIntersect && bounces <= nrBounces){        //Intersection With New Object

        gPoint = add3( mul3c(ray,gDist), prevPoint);   //3D Point of Intersection
        rgb = mul3c (getColor(rgb,gType,gIndex), 1.0/sqrt((float)bounces));

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


void display()
{
  
  emitPhotons();
	
  // run CUDA kernel
  runCuda();
  
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