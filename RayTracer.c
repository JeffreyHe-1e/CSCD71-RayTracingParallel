/*
  CSC D18 - RayTracer code.

  Written Dec. 9 2010 - Jan 20, 2011 by F. J. Estrada
  Freely distributable for academic purposes only.

  Uses Tom F. El-Maraghi's code for computing inverse
  matrices. You will need to compile together with
  svdDynamic.c

  You need to understand the code provided in
  this file, the corresponding header file, and the
  utils.c and utils.h files. Do not worry about
  svdDynamic.c, we need it only to compute
  inverse matrices.

  You only need to modify or add code in sections
  clearly marked "TO DO" - remember to check what
  functionality is actually needed for the corresponding
  assignment!

  Last updated: Aug. 2017   - F.J.E.
*/

#include "utils.h" // <-- This includes RayTracer.h

// A couple of global structures and data: An object list, a light list, and the
// maximum recursion depth
struct object3D *object_list;
struct pointLS *light_list;
struct textureNode *texture_list;
int MAX_DEPTH;
struct view *cam; // Camera and view for this scene

// Set up background colour here
struct colourRGB background = {.R = 0.6, .G = 0.8, .B = 0.9}; // Background colour

#include "buildscene.c" // Import scene definition

void rtShade(struct object3D *obj, struct point3D *p, struct point3D *n, struct ray3D *ray, int depth, double a, double b, struct colourRGB *col)
{
  // This function implements the shading model as described in lecture. It takes
  // - A pointer to the first object intersected by the ray (to get the colour properties)
  // - The coordinates of the intersection point (in world coordinates)
  // - The normal at the point
  // - The ray (needed to determine the reflection direction to use for the global component, as well as for
  //   the Phong specular component)
  // - The current recursion depth
  // - The (a,b) texture coordinates (meaningless unless texture is enabled)
  //
  // Returns:
  // - The colour for this ray (using the col pointer)
  //

  struct colourRGB tmp_col; // Accumulator for colour components
  double R, G, B;           // Colour for the object in R G and B

  // This will hold the colour as we process all the components of
  // the Phong illumination model
  tmp_col.R = 0;
  tmp_col.G = 0;
  tmp_col.B = 0;

  // texture map ///////////////////////////////////////////////////////////////////
  if (obj->texImg == NULL) // Not textured, use object colour
  {
    R = obj->col.R;
    G = obj->col.G;
    B = obj->col.B;
  }
  else
  {
    // preset the object to colour in case the intersection point is not texture mapped
    // for example: cylinder caps.
    R = obj->col.R;
    G = obj->col.G;
    B = obj->col.B;

    // Get object colour from the texture given the texture coordinates (a,b), and the texturing function
    // for the object. Note that we will use textures also for Photon Mapping.
    obj->textureMap(obj->texImg, a, b, &R, &G, &B);
  }

  // alpha map ///////////////////////////////////////////////////////////////////
  double alpha;
  if (obj->alphaMapped == 1 && obj->alphaMap != NULL)
  { // is alpha mapped, use alpha map.
    alpha = obj->alpha;
    alphaMap(obj->alphaMap, a, b, &alpha);
  }
  else
  {
    alpha = obj->alpha;
  }

  // local variables ///////////////////////////////////////////////////////////////////
  struct pointLS *light;
  struct ray3D srcRay; // s
  struct ray3D reflRay;
  struct point3D specVec;
  // struct point3D camVec; // unused
  struct ray3D refrRay;

  double lambda;
  struct object3D *isectObj;
  struct point3D isectPt;
  struct point3D isectN;
  double isectA, isectB;

  tmp_col.R = 0;
  tmp_col.G = 0;
  tmp_col.B = 0;

  // AMBIENT COMPONENT ///////////////////////////////////////////////////////////////////
  // assume I_a = 1
  tmp_col.R += R * obj->alb.ra;
  tmp_col.G += G * obj->alb.ra;
  tmp_col.B += B * obj->alb.ra;

  // get length of light list
  light = light_list;
  int lightCount = 0;
  while (light != NULL)
  {
    lightCount++;
    light = light->next;
  }

  light = light_list;
  while (light != NULL)
  {
    // calculate ray to light source (don't normalize direction for shadow casting)
    initRay(&srcRay, p, &(ray->d));
    srcRay.d.px = light->p0.px - p->px;
    srcRay.d.py = light->p0.py - p->py;
    srcRay.d.pz = light->p0.pz - p->pz;

    // get vector from lightsource to intersection point
    struct ray3D fromLight;
    initRay(&fromLight, &(light->p0), &(srcRay.d));
    fromLight.d.px = -fromLight.d.px;
    fromLight.d.py = -fromLight.d.py;
    fromLight.d.pz = -fromLight.d.pz;

    // check for shadow by shooting ray from light source
    findFirstHit(&fromLight, &lambda, obj, &isectObj, &isectPt, &isectN, &isectA, &isectB);

    normalize(n);
    normalize(&(srcRay.d));

    if (0 < lambda && lambda < 1)
    {
      // object obstructing light => just ambient light
    }
    else
    { // no obstruction => full phong

      // DIFFUSE COMPONENT ///////////////////////////////////////////////////////////////////
      double diffuseMaxTerm;

      // if frontAndBack flag is set, colour both sides of the object
      if (obj->frontAndBack == 1)
      {
        diffuseMaxTerm = abs(dot(n, &(srcRay.d)));
      }
      else
      {
        diffuseMaxTerm = max(0, dot(n, &(srcRay.d)));
      }

      tmp_col.R += R * obj->alb.rd * light->col.R * diffuseMaxTerm;
      tmp_col.G += G * obj->alb.rd * light->col.G * diffuseMaxTerm;
      tmp_col.B += B * obj->alb.rd * light->col.B * diffuseMaxTerm;

      // SPECULAR COMPONENT ///////////////////////////////////////////////////////////////////
      struct ray3D c;
      initRay(&c, p, &(ray->d));
      c.d.px = -c.d.px;
      c.d.py = -c.d.py;
      c.d.pz = -c.d.pz;
      normalize(&(c.d));

      // calculate specular vector, which is the reflection of the light source on the normal
      specVec.px = 2 * dot(n, &(srcRay.d)) * n->px - srcRay.d.px;
      specVec.py = 2 * dot(n, &(srcRay.d)) * n->py - srcRay.d.py;
      specVec.pz = 2 * dot(n, &(srcRay.d)) * n->pz - srcRay.d.pz;
      normalize(&specVec);

      // if light source is behind the object, don't calculate specular component
      if (dot(n, &(srcRay.d)) >= 0 || obj->frontAndBack == 1)
      {
        tmp_col.R += obj->alb.rs * light->col.R * pow(max(0, dot(&c.d, &specVec)), obj->shinyness);
        tmp_col.G += obj->alb.rs * light->col.G * pow(max(0, dot(&c.d, &specVec)), obj->shinyness);
        tmp_col.B += obj->alb.rs * light->col.B * pow(max(0, dot(&c.d, &specVec)), obj->shinyness);
      }
    }
    light = light->next;
  } // end of light loop

  // multiply by alpha
  tmp_col.R *= alpha;
  tmp_col.G *= alpha;
  tmp_col.B *= alpha;

  // REFLECTED & REFRACTED (GLOBAL) COMPONENT ///////////////////////////////////////////////////////////////////
  if (depth < MAX_DEPTH)
  {
    // calculate reflection ray
    struct colourRGB reflCol;
    initRay(&reflRay, p, &(ray->d));
    ray->d.px = -ray->d.px;
    ray->d.py = -ray->d.py;
    ray->d.pz = -ray->d.pz;
    reflRay.d.px = 2 * dot(n, &(ray->d)) * n->px - ray->d.px;
    reflRay.d.py = 2 * dot(n, &(ray->d)) * n->py - ray->d.py;
    reflRay.d.pz = 2 * dot(n, &(ray->d)) * n->pz - ray->d.pz;
    normalize(&(reflRay.d));

    // ray trace
    rayTrace(&reflRay, depth + 1, &reflCol, obj);

    // global component is r_g * I_g, independent of alpha
    tmp_col.R += obj->alb.rg * reflCol.R;
    tmp_col.G += obj->alb.rg * reflCol.G;
    tmp_col.B += obj->alb.rg * reflCol.B;

    // if alpha < 1, add refraction
    if (alpha < 1)
    {
      struct colourRGB refrCol;
      double dotProd = dot(n, &(ray->d)); // cosine of angle between the ray and the normal

      double n1 = ray->r_index;  // refractive index of the material the incoming ray is traveling through
      double n2 = obj->r_index;  // refractive index of the material the incoming ray is entering
      struct point3D refrVec;    // refraction (transmitted) ray's direction vector
      struct point3D refrOrigin; // refraction (transmitted) ray's origin point p0
      getRefraction(&(ray->d), n, n1, n2, &(refrVec));
      // printf("[debug] n1 n2, %f %f\n", n1, n2);

      // schlick's approximation
      // TODO this might not be accurate. It depends on if the ray is traveling into the object or out of it.
      double R0 = pow((n1 - n2) / (n1 + n2), 2);
      double Rtheta = R0 + (1 - R0) * pow((1 - dotProd), 5);

      // transmitted intensity
      double r_t = 1 - Rtheta;

      initRay(&refrRay, p, &(refrVec));

      // update incoming ray refractive index
      // if obj normal is in the same direction as the ray, then the ray is inside the object
      if (dotProd < 0)
      { // ray is exiting the object
        refrRay.r_index = n1 / n2;
      }
      else
      { // ray is entering the object
        refrRay.r_index = n1 * n2;
      }

      // printf("refrRay.r_index: %f\n", refrRay.r_index);

      // d_t = rb + (rc - sqrt(1 - r^2 * (1 - c^2))) * n
      // where...
      // b is the direction of the ray
      // c = -n dot b
      // r is the ratio of the indices of refraction c_2/c_1 = n_1/n_2
      struct point3D b = ray->d;
      double c = -dot(n, &b);
      double r = n1 / n2;
      normalize(n);
      normalize(&b);
      refrRay.d.px = r * b.px + (r * c - sqrt(1 - pow(r, 2) * (1 - pow(c, 2)))) * n->px;
      refrRay.d.py = r * b.py + (r * c - sqrt(1 - pow(r, 2) * (1 - pow(c, 2)))) * n->py;
      refrRay.d.pz = r * b.pz + (r * c - sqrt(1 - pow(r, 2) * (1 - pow(c, 2)))) * n->pz;
      refrRay.d.pw = 0;

      normalize(&(refrRay.d));

      // walk the ray by a little to ensure the origin point is not so close to the object's surface
      // this is done to eliminate numerical errors.
      rayPosition(&refrRay, 0.001, &refrOrigin);
      refrRay.p0.px = refrOrigin.px;
      refrRay.p0.py = refrOrigin.py;
      refrRay.p0.pz = refrOrigin.pz;
      refrRay.p0.pw = 1;

      rayTrace(&refrRay, depth + 1, &refrCol, NULL);

      // refracted component is (1-alpha) * r_t * I_t
      if (refrCol.R >= 0)
      {
        // printf("refrCol: %f %f %f\n", refrCol.R, refrCol.G, refrCol.B);
        tmp_col.R += (1 - alpha) * r_t * refrCol.R;
        tmp_col.G += (1 - alpha) * r_t * refrCol.G;
        tmp_col.B += (1 - alpha) * r_t * refrCol.B;

        // DEBUG RECRSION
        // if (depth >= 2)
        // {
        //   tmp_col.R = 1;
        //   tmp_col.G = 0;
        //   tmp_col.B = 0;
        // }
      }
    }
  } // end if (depth < MAX_DEPTH)

  // [DEBUG] normals
//    tmp_col.R = (n->px +1)/2;
//    tmp_col.G = (n->py +1)/2;
//    tmp_col.B = (n->pz +1)/2;

  // [DEBUG] texture coordinate maps
//    tmp_col.R = a >= 0 || b >= 0 ? a : 0.5;
//    tmp_col.G = a >= 0 || b >= 0 ? b : 0.5;
//    tmp_col.B = a >= 0 || b >= 0 ? 0 : 0.5;

  // output colours
  col->R = tmp_col.R > 1 ? 1 : tmp_col.R;
  col->G = tmp_col.G > 1 ? 1 : tmp_col.G;
  col->B = tmp_col.B > 1 ? 1 : tmp_col.B;

  // Be sure to update 'col' with the final colour computed here!
  return;
}

void findFirstHit(struct ray3D *ray, double *lambda, struct object3D *Os, struct object3D **obj, struct point3D *p, struct point3D *n, double *a, double *b)
{
  // Find the closest intersection between the ray and any objects in the scene.
  // Inputs:
  //   *ray    -  A pointer to the ray being traced
  //   *Os     -  'Object source' is a pointer toward the object from which the ray originates. It is used for reflected or refracted rays
  //              so that you can check for and ignore self-intersections as needed. It is NULL for rays originating at the center of
  //              projection
  // Outputs:
  //   *lambda -  A pointer toward a double variable 'lambda' used to return the lambda at the intersection point
  //   **obj   -  A pointer toward an (object3D *) variable so you can return a pointer to the object that has the closest intersection with
  //              this ray (this is required so you can do the shading)
  //   *p      -  A pointer to a 3D point structure so you can store the coordinates of the intersection point
  //   *n      -  A pointer to a 3D point structure so you can return the normal at the intersection point
  //   *a, *b  -  Pointers toward double variables so you can return the texture coordinates a,b at the intersection point

  struct object3D *curObj = object_list;

  double curLambda;
  struct point3D curP;
  struct point3D curN;
  double curA;
  double curB;

  *lambda = -1;
  while (curObj != NULL)
  {
    // don't check for object self intersection
    if (curObj == Os)
    {
      curObj = curObj->next;
      continue;
    }

    // if (curObj->isLightSource == 1)
    // {
    // curObj = curObj->next;
    // continue;
    // }

    // find lambda of intersection
    curObj->intersect(curObj, ray, &curLambda, &curP, &curN, &curA, &curB);

    // update the closest lambda
    if ((*lambda < 0 || curLambda < *lambda) && curLambda > 0)
    {
      *lambda = curLambda;
      *p = curP;
      *n = curN;
      *a = curA;
      *b = curB;
      *obj = curObj;
    }

    curObj = curObj->next;
  }
}

void rayTrace(struct ray3D *ray, int depth, struct colourRGB *col, struct object3D *Os)
{
  // Trace one ray through the scene.
  //
  // Parameters:
  //   *ray   -  A pointer to the ray being traced
  //   depth  -  Current recursion depth for recursive raytracing
  //   *col   - Pointer to an RGB colour structure so you can return the object colour
  //            at the intersection point of this ray with the closest scene object.
  //   *Os    - 'Object source' is a pointer to the object from which the ray
  //            originates so you can discard self-intersections due to numerical
  //            errors. NULL for rays originating from the center of projection.

  double lambda;        // Lambda at intersection
  double a, b;          // Texture coordinates
  struct object3D *obj; // Pointer to object at intersection
  struct point3D p;     // Intersection point
  struct point3D n;     // Normal at intersection
  struct colourRGB I;   // Colour returned by shading function

  if (depth > MAX_DEPTH) // Max recursion depth reached. Return invalid colour.
  {
    col->R = -1;
    col->G = -1;
    col->B = -1;
    return;
  }

  // Find first hit
  findFirstHit(ray, &lambda, Os, &obj, &p, &n, &a, &b);

  // No intersection, return background colour
  if (lambda < 0)
  {
    col->R = background.R;
    col->G = background.G;
    col->B = background.B;
    return;
  }

  // Shade the point hit
  rtShade(obj, &p, &n, ray, depth, a, b, &I);

  // Set the colour to the shaded colour
  col->R = I.R;
  col->G = I.G;
  col->B = I.B;
}

int main(int argc, char *argv[])
{
  // Main function for the raytracer. Parses input parameters,
  // sets up the initial blank image, and calls the functions
  // that set up the scene and do the raytracing.
  struct image *im;       // Will hold the raytraced image
  int sx;                 // Size of the raytraced image
  int antialiasing;       // Flag to determine whether antialiaing is enabled or disabled
  char output_name[1024]; // Name of the output file for the raytraced .ppm image
  struct point3D e;       // Camera view parameters 'e', 'g', and 'up'
  struct point3D g;
  struct point3D up;
  double du, dv;               // Increase along u and v directions for pixel coordinates
  struct point3D pc, d;        // Point structures to keep the coordinates of a pixel and
                               // the direction or a ray
  struct ray3D ray;            // Structure to keep the ray from e to a pixel
  struct colourRGB col;        // Return colour for raytraced pixels
  int i, j;                    // Counters for pixel coordinates
  unsigned char *rgbIm;

  if (argc < 5)
  {
    fprintf(stderr, "RayTracer: Can not parse input parameters\n");
    fprintf(stderr, "USAGE: RayTracer size rec_depth antialias output_name\n");
    fprintf(stderr, "   size = Image size (both along x and y)\n");
    fprintf(stderr, "   rec_depth = Recursion depth\n");
    fprintf(stderr, "   antialias = A single digit, 0 disables antialiasing. Anything else enables antialiasing\n");
    fprintf(stderr, "   output_name = Name of the output file, e.g. MyRender.ppm\n");
    exit(0);
  }
  sx = atoi(argv[1]);
  MAX_DEPTH = atoi(argv[2]);
  if (atoi(argv[3]) == 0)
    antialiasing = 0;
  else
    antialiasing = 1;
  strcpy(&output_name[0], argv[4]);

  fprintf(stderr, "Rendering image at %d x %d\n", sx, sx);
  fprintf(stderr, "Recursion depth = %d\n", MAX_DEPTH);
  if (!antialiasing)
    fprintf(stderr, "Antialising is off\n");
  else
    fprintf(stderr, "Antialising is on\n");
  fprintf(stderr, "Output file name: %s\n", output_name);

  object_list = NULL;
  light_list = NULL;
  texture_list = NULL;

  // Allocate memory for the new image
  im = newImage(sx, sx);
  if (!im)
  {
    fprintf(stderr, "Unable to allocate memory for raytraced image\n");
    exit(0);
  }
  else
    rgbIm = (unsigned char *)im->rgbdata;

  // Mind the homogeneous coordinate w of all vectors below. DO NOT
  // forget to set it to 1, or you'll get junk out of the
  // geometric transformations later on.

  // Camera center is at (0,0,-1)
  e.px = 0;
  e.py = 0;
  e.pz = -3;
  e.pw = 1;

  // To define the gaze vector, we choose a point 'pc' in the scene that
  // the camera is looking at, and do the vector subtraction pc-e.
  // Here we set up the camera to be looking at the origin.
  g.px = 0 - e.px;
  g.py = 0 - e.py;
  g.pz = 0 - e.pz;
  g.pw = 1;
  // In this case, the camera is looking along the world Z axis, so
  // vector w should end up being [0, 0, -1]

  // Define the 'up' vector to be the Y axis
  up.px = 0;
  up.py = 1;
  up.pz = 0;
  up.pw = 1;

  // Set up view with given the above vectors, a 4x4 window,
  // and a focal length of -1 (why? where is the image plane?)
  // Note that the top-left corner of the window is at (-2, 2)
  // in camera coordinates.
  //  cam = setupView(&e, &g, &up, -1, -2, 2, 4);   // the original camera position

  cam = setupView(&e, &g, &up, -1, -1, 1, 2);

  time_t sec;
  time(&sec);
  printf("complete initialization: %ld\n", sec);

  // set up scene (may or may not include camera setup).
  buildScene(); // Create a scene. This defines all the
                // objects in the world of the raytracer

  time(&sec);
  printf("complete buildscene: %ld\n", sec);

  if (cam == NULL)
  {
    fprintf(stderr, "Unable to set up the view and camera parameters. Out of memory!\n");
    cleanup(object_list, light_list, texture_list);
    deleteImage(im);
    exit(0);
  }

  // Do the raytracing
  //////////////////////////////////////////////////////
  // TO DO: You will need code here to do the raytracing
  //        for each pixel in the image. Refer to the
  //        lecture notes, in particular, to the
  //        raytracing pseudocode, for details on what
  //        to do here. Make sure you understand the
  //        overall procedure of raytracing for a single
  //        pixel.
  //////////////////////////////////////////////////////
  du = cam->wsize / (sx - 1);  // du and dv. In the notes in terms of wl and wr, wt and wb,
  dv = -cam->wsize / (sx - 1); // here we use wl, wt, and wsize. du=dv since the image is
                               // and dv is negative since y increases downward in pixel
                               // coordinates and upward in camera coordinates.

  fprintf(stderr, "View parameters:\n");
  fprintf(stderr, "Left=%f, Top=%f, Width=%f, f=%f\n", cam->wl, cam->wt, cam->wsize, cam->f);
  fprintf(stderr, "Camera to world conversion matrix (make sure it makes sense!):\n");
  printmatrix(cam->C2W);
  fprintf(stderr, "World to camera conversion matrix:\n");
  printmatrix(cam->W2C);
  fprintf(stderr, "\n");

  // fprintf(stderr, "Rendering row: ");
  fprintf(stderr, "Rendering...");

#pragma omp parallel for collapse(2) schedule(dynamic, 10) private(col)
  for (j = 0; j < sx; j++) // For each of the pixels in the image
  {
    for (i = 0; i < sx; i++)
    {
      if (!antialiasing)
      {
        calculatePixel(du, dv, cam, i, j, &col);
      }
      else
      {
        const int NUM_SAMPLES = 10;
        struct colourRGB samples[NUM_SAMPLES];

        // calculate the colour for each sample
        for (int s = 0; s < NUM_SAMPLES; s++)
        {
          // add between 0 and 1 to i and j
          double sample_i = i + drand48();
          double sample_j = j + drand48();

          calculatePixel(du, dv, cam, sample_i, sample_j, &samples[s]);
        }

        // average the colours
        col.R = 0;
        col.G = 0;
        col.B = 0;

        for (int s = 0; s < NUM_SAMPLES; s++)
        {
          col.R += samples[s].R;
          col.G += samples[s].G;
          col.B += samples[s].B;
        }

        col.R /= NUM_SAMPLES;
        col.G /= NUM_SAMPLES;
        col.B /= NUM_SAMPLES;
      }

      // Step 3: set i,j pixel colour to col
      ((unsigned char *)(im->rgbdata))[(j * sx + i) * 3 + 0] = (unsigned char)(255 * col.R);
      ((unsigned char *)(im->rgbdata))[(j * sx + i) * 3 + 1] = (unsigned char)(255 * col.G);
      ((unsigned char *)(im->rgbdata))[(j * sx + i) * 3 + 2] = (unsigned char)(255 * col.B);
    } // end for i
  } // end for j

  fprintf(stderr, "\nDone!\n");
  time(&sec);
  printf("complete render: %ld\n", sec);

  // Output rendered image
  imageOutput(im, output_name);

  // Exit section. Clean up and return.
  cleanup(object_list, light_list, texture_list); // Object, light, and texture lists
  deleteImage(im);                                // Rendered image
  free(cam);                                      // camera view

  time(&sec);
  printf("complete output: %ld\n", sec);
  exit(0);
}

void calculatePixel(double du, double dv, struct view *cam, double i, double j, struct colourRGB *col)
{
  // IMPORTANT! set the direction vector's homogeneous component to `0`
  // so that it isn't affected by the translation component of transforms.

  // Step 1: compute (in world coordinates) rij = pij + lambda*dij where dij = pij - e
  struct ray3D rij;
  struct point3D pij;
  struct point3D dij;

  // calculate pij in camera coordinates
  pij.px = i * du + cam->wl;
  pij.py = j * dv + cam->wt;
  pij.pz = cam->f;
  pij.pw = 1;

  // convert pij to world coordinates
  matVecMult(cam->C2W, &pij);

  // calculate dij in world coordinates
  dij.px = pij.px - cam->e.px;
  dij.py = pij.py - cam->e.py;
  dij.pz = pij.pz - cam->e.pz;

  // set rij
  initRay(&rij, &pij, &dij);

  // Step 2: rayTrace(rij, 1, col, NULL)
  rayTrace(&rij, 1, col, NULL);
}

/**
 * Calculate the direction vector of the refracted ray.
 * In case of total internal refraction, no refraction. Output the zero vector instead.
 * @param incidentVec direction vector of the ray.
 * @param normalVec normal vector of the object the ray hits (assume is normalized).
 * @param nEnv refractive index of the environment the ray is traveling through
 * @param nObj refractive index of the object the ray hits.
 * @param refrVec [out] the refracted ray.
 */
void getRefraction(struct point3D *incidentVec, struct point3D *normalVec, double nEnv, double nObj, struct point3D *refrVec)
{
  /*
   * math source: https://www.scratchapixel.com/lessons/3d-basic-rendering/introduction-to-shading/reflection-refraction-fresnel.html
   * see also: https://graphics.stanford.edu/courses/cs148-10-summer/docs/2006--degreve--reflection_refraction.pdf
   */
  double refrRatio;                // n1 / n2
  struct point3D N;                // the normal vector pointing towards the source of the ray, used for calculations
  struct point3D I = *incidentVec; // the incident ray direction vector
  normalize(&I);
  double dotProd = dot(&I, normalVec);

  if (dotProd < 0)
  { // the ray is entering the object
    refrRatio = nEnv / nObj;
    N = *normalVec;
  }
  else
  { // the ray is exiting the object
    refrRatio = nObj / nEnv;
    N.px = -normalVec->px;
    N.py = -normalVec->py;
    N.pz = -normalVec->pz;
    N.pw = 0;           // flip the normal
    dotProd = -dotProd; // flip the sign of the dot product (consequence of flipping the normal)
  }

  /*
   * magic math:
   * refracted  =  tangent * sin(theta2)  +  normal * cos(theta2)
   * Use Snell's law and geometry to simplify "tangent * sin(theta2)" to:
   *     refrRatio*(I + dotProd*N)
   * Use Snell's law and Pythagorean trig identity, express "cos(theta2)" as:
   *     sqrt( 1 - refrRatio^2 * (1 - dotProd^2) )
   */

  double cosTheta2Squared = 1 - refrRatio * refrRatio * (1 - dotProd * dotProd);
  if (cosTheta2Squared < 0)
  { // total internal reflection, can't take square root of negative number.
    refrVec->px = 0;
    refrVec->py = 0;
    refrVec->pz = 0;
    refrVec->pw = 0;
    return;
  }
  double cosTheta2 = sqrt(cosTheta2Squared);

  normalize(normalVec);

  // refracted  =  tangent * sin(theta2)  +  normal * cos(theta2)
  // equivalently: refracted  =  (refrRatio)I  +  (refrRatio*dotProd - cosTheta2)N.
  refrVec->px = refrRatio * I.px + (refrRatio * dotProd - cosTheta2) * N.px;
  refrVec->py = refrRatio * I.py + (refrRatio * dotProd - cosTheta2) * N.py;
  refrVec->pz = refrRatio * I.pz + (refrRatio * dotProd - cosTheta2) * N.pz;
  refrVec->pw = 0;

  normalize(refrVec);
}