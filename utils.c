/*
   utils.c - F.J. Estrada, Dec. 9, 2010

   Utilities for the ray tracer. You will need to complete
   some of the functions in this file. Look for the sections
   marked "TO DO". Be sure to read the rest of the file and
   understand how the entire code works.

   HOWEVER: Note that there are a lot of incomplete functions
            that will only be used for the advanced ray tracer!
      So, read the handout carefully and implement only
      the code you need for the corresponding assignment.

   Last updated: Aug. 2017  -  F.J.E.
*/

#include "utils.h"

int debugFlag = 0; // [DEBUG] for printing something once.

// A useful 4x4 identity matrix which can be used at any point to
// initialize or reset object transformations
double eye4x4[4][4] = {{1.0, 0.0, 0.0, 0.0},
                       {0.0, 1.0, 0.0, 0.0},
                       {0.0, 0.0, 1.0, 0.0},
                       {0.0, 0.0, 0.0, 1.0}};

/////////////////////////////////////////////
// Primitive data structure section
/////////////////////////////////////////////
struct point3D *newPoint(double px, double py, double pz)
{
  // Allocate a new point structure, initialize it to
  // the specified coordinates, and return a pointer
  // to it.

  struct point3D *pt = (struct point3D *)calloc(1, sizeof(struct point3D));
  if (!pt)
    fprintf(stderr, "Out of memory allocating point structure!\n");
  else
  {
    pt->px = px;
    pt->py = py;
    pt->pz = pz;
    pt->pw = 1.0;
  }
  return (pt);
}

struct pointLS *newPLS(struct point3D *p0, double r, double g, double b)
{
  // Allocate a new point light sourse structure. Initialize the light
  // source to the specified RGB colour
  // Note that this is a point light source in that it is a single point
  // in space, if you also want a uniform direction for light over the
  // scene (a so-called directional light) you need to place the
  // light source really far away.

  struct pointLS *ls = (struct pointLS *)calloc(1, sizeof(struct pointLS));
  if (!ls)
    fprintf(stderr, "Out of memory allocating light source!\n");
  else
  {
    memcpy(&ls->p0, p0, sizeof(struct point3D)); // Copy light source location

    ls->col.R = r; // Store light source colour and
    ls->col.G = g; // intensity
    ls->col.B = b;
  }
  return (ls);
}

// ========================= //
// Ray and normal transforms //
// ========================= //

/**
 * Transforms a ray using the inverse transform for the specified object.
 * This is so that we can use the intersection test for the canonical object.
 * Note that this has to be done carefully!
 * Note that the transformed direction vector is NOT normalized!
 * @param ray_orig        The ray to be transformed.
 * @param ray_transformed [out] The transformed ray.
 * @param obj             Object to get the transform from.
 */
inline void rayTransform(struct ray3D *ray_orig, struct ray3D *ray_transformed, struct object3D *obj)
{
  *ray_transformed = *ray_orig;

  // transform the ray's origin
  matVecMult(obj->Tinv, &(ray_transformed->p0));

  // transform the ray's direction vector
  ray_transformed->d.pw = 0; // set to 0 to avoid translation
  matVecMult(obj->Tinv, &(ray_transformed->d));
}

inline void normalTransform(struct point3D *n_orig, struct point3D *n_transformed, struct object3D *obj)
{
  // Computes the normal at an affinely transformed point given the original normal and the
  // object's inverse transformation. From the notes:
  // n_transformed=A^-T*n normalized.

  /* transpose A^-1*/
  double transTinv[4][4];
  matTrans(obj->Tinv, transTinv);

  *n_transformed = *n_orig;
  matVecMult(transTinv, n_transformed);
}

/////////////////////////////////////////////
// Object management section
/////////////////////////////////////////////
void insertObject(struct object3D *o, struct object3D **list)
{
  if (o == NULL)
    return;
  // Inserts an object into the object list.
  if (*(list) == NULL)
  {
    *(list) = o;
    (*(list))->next = NULL;
  }
  else
  {
    o->next = (*(list))->next;
    (*(list))->next = o;
  }
}

struct object3D *newPlane(double ra, double rd, double rs, double rg, double r, double g, double b, double alpha, double r_index, double shiny)
{
  // Intialize a new plane with the specified parameters:
  // ra, rd, rs, rg - Albedos for the components of the Phong model
  // r, g, b, - Colour for this plane
  // alpha - Transparency, must be set to 1 unless you are doing refraction
  // r_index - Refraction index if you are doing refraction.
  // shiny - Exponent for the specular component of the Phong model
  //
  // The plane is defined by the following vertices (CCW)
  // (1,1,0), (-1,1,0), (-1,-1,0), (1,-1,0)
  // With normal vector (0,0,1) (i.e. parallel to the XY plane)

  struct object3D *plane = (struct object3D *)calloc(1, sizeof(struct object3D));

  if (!plane)
    fprintf(stderr, "Unable to allocate new plane, out of memory!\n");
  else
  {
    plane->alb.ra = ra;
    plane->alb.rd = rd;
    plane->alb.rs = rs;
    plane->alb.rg = rg;
    plane->col.R = r;
    plane->col.G = g;
    plane->col.B = b;
    plane->alpha = alpha;
    plane->r_index = r_index;
    plane->shinyness = shiny;
    plane->intersect = &planeIntersect;
    plane->surfaceCoords = &planeCoordinates;
    plane->randomPoint = &planeSample;
    plane->texImg = NULL;
    plane->photonMap = NULL;
    plane->normalMap = NULL;
    memcpy(&plane->T[0][0], &eye4x4[0][0], 16 * sizeof(double));
    memcpy(&plane->Tinv[0][0], &eye4x4[0][0], 16 * sizeof(double));
    plane->textureMap = &texMap;
    plane->frontAndBack = 1;
    plane->photonMapped = 0;
    plane->normalMapped = 0;
    plane->isCSG = 0;
    plane->isLightSource = 0;
    plane->CSGnext = NULL;
    plane->next = NULL;
  }
  return (plane);
}

struct object3D *newSphere(double ra, double rd, double rs, double rg, double r, double g, double b, double alpha, double r_index, double shiny)
{
  // Intialize a new sphere with the specified parameters:
  // ra, rd, rs, rg - Albedos for the components of the Phong model
  // r, g, b, - Colour for this plane
  // alpha - Transparency, must be set to 1 unless you are doing refraction
  // r_index - Refraction index if you are doing refraction.
  // shiny -Exponent for the specular component of the Phong model
  //
  // This is assumed to represent a unit sphere centered at the origin.
  //

  struct object3D *sphere = (struct object3D *)calloc(1, sizeof(struct object3D));

  if (!sphere)
    fprintf(stderr, "Unable to allocate new sphere, out of memory!\n");
  else
  {
    sphere->alb.ra = ra;
    sphere->alb.rd = rd;
    sphere->alb.rs = rs;
    sphere->alb.rg = rg;
    sphere->col.R = r;
    sphere->col.G = g;
    sphere->col.B = b;
    sphere->alpha = alpha;
    sphere->r_index = r_index;
    sphere->shinyness = shiny;
    sphere->intersect = &sphereIntersect;
    sphere->surfaceCoords = &sphereCoordinates;
    sphere->randomPoint = &sphereSample;
    sphere->texImg = NULL;
    sphere->photonMap = NULL;
    sphere->normalMap = NULL;
    memcpy(&sphere->T[0][0], &eye4x4[0][0], 16 * sizeof(double));
    memcpy(&sphere->Tinv[0][0], &eye4x4[0][0], 16 * sizeof(double));
    sphere->textureMap = &texMap;
    sphere->frontAndBack = 0;
    sphere->photonMapped = 0;
    sphere->normalMapped = 0;
    sphere->isCSG = 0;
    sphere->isLightSource = 0;
    sphere->CSGnext = NULL;
    sphere->next = NULL;
  }
  return (sphere);
}

/**
 * Initialize a new canonical cylinder with the specified parameters.
 * The canonical cylinder:
 *   radius = 1,
 *   height = 1,
 *   aligned with the z axis,
 *   base sits on the xy plane.
 * @param ra, rd, rs, rg Albedos for the components of the Phong model
 * @param r, g, b Colour for this cylinder
 * @param alpha Transparency
 * @param R_index refractive index
 * @param shiny Exponent for the specular component of the Phong model
 * @return
 */
struct object3D *newCyl(double ra, double rd, double rs, double rg, double r, double g, double b, double alpha, double r_index, double shiny)
{
  struct object3D *cylinder = (struct object3D *)calloc(1, sizeof(struct object3D));

  if (!cylinder)
    fprintf(stderr, "Unable to allocate new cylinder, out of memory!\n");
  else
  {
    cylinder->alb.ra = ra;
    cylinder->alb.rd = rd;
    cylinder->alb.rs = rs;
    cylinder->alb.rg = rg;
    cylinder->col.R = r;
    cylinder->col.G = g;
    cylinder->col.B = b;
    cylinder->alpha = alpha;
    cylinder->r_index = r_index;
    cylinder->shinyness = shiny;
    cylinder->intersect = &cylIntersect;
    cylinder->surfaceCoords = &cylCoordinates;
    cylinder->randomPoint = &cylSample;
    cylinder->texImg = NULL;
    cylinder->photonMap = NULL;
    cylinder->normalMap = NULL;
    memcpy(&cylinder->T[0][0], &eye4x4[0][0], 16 * sizeof(double));
    memcpy(&cylinder->Tinv[0][0], &eye4x4[0][0], 16 * sizeof(double));
    cylinder->textureMap = &texMap;
    cylinder->frontAndBack = 0;
    cylinder->photonMapped = 0;
    cylinder->normalMapped = 0;
    cylinder->isCSG = 0;
    cylinder->isLightSource = 0;
    cylinder->CSGnext = NULL;
    cylinder->next = NULL;
  }
  return (cylinder);
}

/**
 * Initialize a new canonical cone with the specified parameters.
 * The canonical cone:
 *   radius = 1,
 *   height = 1,
 *   the apex (point) is aligned at the origin
 *   the base is unit circle, parallel with the xy plane at z = 1
 * @param ra, rd, rs, rg Albedos for the components of the Phong model
 * @param r, g, b Colour for this cone
 * @param alpha Transparency
 * @param R_index refractive index
 * @param shiny Exponent for the specular component of the Phong model
 * @return
 */
struct object3D *newCone(double ra, double rd, double rs, double rg, double r, double g, double b, double alpha, double r_index, double shiny)
{
  struct object3D *cone = (struct object3D *)calloc(1, sizeof(struct object3D));

  if (!cone)
    fprintf(stderr, "Unable to allocate new cone, out of memory!\n");
  else
  {
    cone->alb.ra = ra;
    cone->alb.rd = rd;
    cone->alb.rs = rs;
    cone->alb.rg = rg;
    cone->col.R = r;
    cone->col.G = g;
    cone->col.B = b;
    cone->alpha = alpha;
    cone->r_index = r_index;
    cone->shinyness = shiny;
    cone->intersect = &coneIntersect;
    cone->surfaceCoords = &coneCoordinates;
    cone->randomPoint = &cylSample;
    cone->texImg = NULL;
    cone->photonMap = NULL;
    cone->normalMap = NULL;
    memcpy(&cone->T[0][0], &eye4x4[0][0], 16 * sizeof(double));
    memcpy(&cone->Tinv[0][0], &eye4x4[0][0], 16 * sizeof(double));
    cone->textureMap = &texMap;
    cone->frontAndBack = 0;
    cone->photonMapped = 0;
    cone->normalMapped = 0;
    cone->isCSG = 0;
    cone->isLightSource = 0;
    cone->CSGnext = NULL;
    cone->next = NULL;
  }
  return (cone);
}

struct object3D *newCube(double ra, double rd, double rs, double rg, double r, double g, double b, double alpha, double r_index, double shiny)
{
  // Intialize a new cube with the specified parameters:
  // ra, rd, rs, rg - Albedos for the components of the Phong model
  // r, g, b, - Colour for this plane
  // alpha - Transparency, must be set to 1 unless you are doing refraction
  // r_index - Refraction index if you are doing refraction.
  // shiny -Exponent for the specular component of the Phong model
  //
  // This is assumed to represent a unit cube centered at the origin.
  //

  struct object3D *cube = (struct object3D *)calloc(1, sizeof(struct object3D));

  if (!cube)
    fprintf(stderr, "Unable to allocate new cube, out of memory!\n");
  else
  {
    cube->alb.ra = ra;
    cube->alb.rd = rd;
    cube->alb.rs = rs;
    cube->alb.rg = rg;
    cube->col.R = r;
    cube->col.G = g;
    cube->col.B = b;
    cube->alpha = alpha;
    cube->r_index = r_index;
    cube->shinyness = shiny;
    cube->intersect = &cubeIntersect;
    cube->surfaceCoords = &cubeCoordinates;
    cube->randomPoint = &cubeSample;
    cube->texImg = NULL;
    cube->photonMap = NULL;
    cube->normalMap = NULL;
    memcpy(&cube->T[0][0], &eye4x4[0][0], 16 * sizeof(double));
    memcpy(&cube->Tinv[0][0], &eye4x4[0][0], 16 * sizeof(double));
    cube->textureMap = &texMap;
    cube->frontAndBack = 0;
    cube->photonMapped = 0;
    cube->normalMapped = 0;
    cube->isCSG = 0;
    cube->isLightSource = 0;
    cube->CSGnext = NULL;
    cube->next = NULL;
  }
  return (cube);
}

///////////////////////////////////////////////////////////////////////////////////////
// TO DO:
//	Complete the functions that compute intersections for the canonical plane
//      and canonical sphere with a given ray. This is the most fundamental component
//      of the raytracer.
///////////////////////////////////////////////////////////////////////////////////////

/**
 * Computes and returns the value of 'lambda' at the intersection
 * between the specified ray and the specified canonical plane.
 *
 * The canonical plane is a square of side length 2, aligned with
 * the x-y plane, and centered at the origin, with normal in the
 * direction of the positive z axis.
 *
 * @param plane   The plane object.
 * @param ray     The ray object in world coordinate frame.
 * @param lambda  [out] The 'lambda' of the ray at the intersection, set to `-1` if no intersection.
 * @param p       [out] The intersection point.
 * @param n       [out] The normal at the intersection.
 * @param a       [out] The 'a' texture coordinate (FOR ADVANCED RAY TRACER).
 * @param b       [out] The 'b' texture coordinate (FOR ADVANCED RAY TRACER).
 */
void planeIntersect(struct object3D *plane, struct ray3D *ray, double *lambda, struct point3D *p, struct point3D *n, double *a, double *b)
{
  // Step 1: Get transformed ray (T^-1 * ray)
  struct ray3D ray_transformed;
  initRay(&ray_transformed, &(ray->p0), &(ray->d));
  rayTransform(ray, &ray_transformed, plane);

  // Step 2: Intersect transformed ray with canonical plane

  // the canonical normal vector of the plane (positive z axis)
  struct point3D canNormal = {.px = 0, .py = 0, .pz = 1, .pw = 0};

  // Check if ray is parallel to the plane
  if (ray_transformed.d.pz == 0)
  {
    *lambda = -1;
    return;
  }

  // Solve for lambda
  *lambda = -ray_transformed.p0.pz / ray_transformed.d.pz;

  // Check if intersection is behind the ray
  if (*lambda < 0)
  {
    *lambda = -1;
    return;
  }

  // Check if transformed ray intersects canonical plane by getting transformed intersection point
  struct point3D p_transformed;
  ray_transformed.rayPos(&ray_transformed, *lambda, &p_transformed);

  if (p_transformed.px < -1 || p_transformed.px > 1 || p_transformed.py < -1 || p_transformed.py > 1)
  {
    *lambda = -1;
    return;
  }

  // Compute intersection point
  ray->rayPos(ray, *lambda, p);

  // Set texture coordinates
  *a = (p_transformed.px + 1) / 2;
  *b = (p_transformed.py + 1) / 2;

  // Compute normal
  // adjust normal according to normal map (if the object is normal mapped)
  if (plane->normalMapped == 1 && plane->normalMap != NULL)
  {
    struct point3D tanVec = {.px = 1, .py = 0, .pz = 0, .pw = 0};   // tangent vector in the 'a' texture coordinate direction (canonical x-axis)
    struct point3D biTanVec = {.px = 0, .py = 1, .pz = 0, .pw = 0}; // bi-tangent vector in the 'b' texture coordinate directionn (canonical y-axis)

    normMap(plane->normalMap, *a, *b, &canNormal, &tanVec, &biTanVec);
    normalize(&canNormal);
  }
  struct point3D normal_transformed;
  normalTransform(&canNormal, &normal_transformed, plane);
  *n = normal_transformed;
}

/**
 * Computes and returns the value of 'lambda' at the intersection
 * between the specified ray and the specified canonical sphere.
 *
 * The canonical sphere has radius 1, and is centered at the origin.
 *
 * @param sphere The sphere object.
 * @param ray    The ray object in world coordinate frame.
 * @param lambda [out] The 'lambda' of the ray at the intersection, set to `-1` if no intersection.
 * @param p      [out] The intersection point.
 * @param n      [out] The normal at the intersection.
 * @param a      [out] The 'a' texture coordinate (value within [0,1]).
 * @param b      [out] The 'b' texture coordinate (value within [0,1]).
 */
void sphereIntersect(struct object3D *sphere, struct ray3D *ray, double *lambda, struct point3D *p, struct point3D *n, double *a, double *b)
{
  /*
   * reference: Wikipedia Line–sphere intersection
   *
   * sphere: centered at o = [0 0 0], radius r = 1
   * ray: p + lambda * d    (d is not normalized because of transforms!)
   *
   * | p + lambda * d |^2  ==  1       , which expands to:
   * lambda^2 (d.d)  +  2 lambda (d.p)  +  (p.p) - 1 ==  0
   *
   * solve with quadratic formula: lambda  =  -B ± sqrt(B^2 - 4AC) / 2A
   * A  =  d.d
   * B  =  2 * d.p
   * C  =  p.p - 1
   */

  // Step 1: Get transformed ray (T^-1 * ray)
  struct ray3D rayT;
  initRay(&rayT, &(ray->p0), &(ray->d));
  rayTransform(ray, &rayT, sphere);

  // Step 2: Get intersection and normal with transformed ray
  double A = dot(&rayT.d, &rayT.d);
  double B = 2 * dot(&rayT.d, &rayT.p0);
  double C = dot(&rayT.p0, &rayT.p0) - 1;

  double discriminant = B * B - 4 * A * C;
  double root1 = -1;
  double root2 = -1;

  if (discriminant > 0)
  { // 2 real roots (root1 < root2)
    root1 = (-B - sqrt(discriminant)) / (2 * A);
    root2 = (-B + sqrt(discriminant)) / (2 * A);

    if (root1 > 0 && root2 > 0)
    { // 2 positive roots
      *lambda = root1;
    }
    else if (root1 < 0 && root2 < 0)
    { // 2 negative roots (no intersection)
      *lambda = -1;
      return;
    }
    else
    { // one positive, one negative (ray is inside the sphere)
      *lambda = root1;
    }
  }
  else
  { // no real roots (including when ray skims the sphere)
    *lambda = -1;
    return;
  }

  struct point3D objP; // intersection point in object coordinates
  rayT.rayPos(&rayT, *lambda, &objP);
  // since the sphere is unit sphere centered at the origin,
  // the normal is the in intersection point.

  // step 3: Compute values in world coordinates
  ray->rayPos(ray, *lambda, p);

  // calculate texture mapping. From Wikipedia: UV mapping
  // a=0=1 is point aligned with negative x-axis
  // b=0 is point at z=-1, b=1 is point at z=1
  *a = 0.5 + atan2(objP.py, objP.px) / (2 * PI); // angle along xy-plane (longitude)
  *b = 0.5 + asin(objP.pz) / PI;                 // angle of elevation from xy-plane (latitude)

  // compute normal
  // adjust normal according to normal map (if the object is normal mapped)
  struct point3D canNormal = {.px = objP.px, .py = objP.py, .pz = objP.pz, .pw = 0};
  if (sphere->normalMapped == 1 && sphere->normalMap != NULL)
  {
    struct point3D up = {.px = 0, .py = 0, .pz = -1, .pw = 0};   // used to calculate tangent and bi-tangent
    struct point3D *tanVec = cross(&up, &canNormal);   // tangent vector in the 'a' texture coordinate direction (parallel to xy-plane)
    struct point3D *biTanVec = cross(&canNormal, tanVec); // bi-tangent vector in the 'b' texture coordinate direction (points up)

    normMap(sphere->normalMap, *a, *b, &canNormal, tanVec, biTanVec);
    normalize(&canNormal);
  }
  struct point3D normal_transformed;
  normalTransform(&canNormal, &normal_transformed, sphere);
  *n = normal_transformed;
}

/**
 * Computes and returns the value of 'lambda' at the intersection
 * between the specified ray and the specified canonical cylinder.
 *
 * The canonical cylinder:
 *   radius = 1,
 *   height = 1,
 *   aligned with the z axis,
 *   base sits on the xy plane and grows towards positive z.
 *
 * @param cylinder The cylinder object.
 * @param ray      The ray object in world coordinate frame.
 * @param lambda   [out] The 'lambda' of the ray at the intersection, set to `-1` if no intersection.
 * @param p        [out] The intersection point.
 * @param n        [out] The normal at the intersection.
 * @param a        [out] The 'a' texture coordinate (FOR ADVANCED RAY TRACER).
 * @param b        [out] The 'b' texture coordinate (FOR ADVANCED RAY TRACER).
 */
void cylIntersect(struct object3D *cylinder, struct ray3D *ray, double *lambda, struct point3D *p, struct point3D *n, double *a, double *b)
{
  /*
   * Reference: PACO's lecture notes
   *
   * cylinder wall: x^2 + y^2   =  1    {0 < z < 1}
   * cylinder caps: x^2 + y^2  <=  1    {z = 0, 1}
   * ray: p + lambda * d    (d is not normalized because of transforms!)
   */

  // Step 1: Get transformed ray (T^-1 * ray) ////////////////////////////////////////////
  struct ray3D rayT;
  initRay(&rayT, &(ray->p0), &(ray->d));
  rayTransform(ray, &rayT, cylinder);

  // Step 2: find intersection //////////////////////////////////////////////////////////
  // lambda candidates. index 0,1: wall intersection. index 2: z=0 cap intersection. index 3: z=1 cap intersection.
  double lambdaCand[4] = {-1, -1, -1, -1};

  // check CYLINDER WALL intersection //
  // just sphere intersection, but we pretend the z component of everything is 0
  double A = rayT.d.px * rayT.d.px + rayT.d.py * rayT.d.py;         // d.d
  double B = 2 * (rayT.p0.px * rayT.d.px + rayT.p0.py * rayT.d.py); // 2 * d.p
  double C = rayT.p0.px * rayT.p0.px + rayT.p0.py * rayT.p0.py - 1; // p.p - 1

  double discriminant = B * B - 4 * A * C;
  struct point3D isectPt; // the intersection point in object coordinates

  if (discriminant > 0)
  {
    lambdaCand[0] = (-B - sqrt(discriminant)) / (2 * A);
    lambdaCand[1] = (-B + sqrt(discriminant)) / (2 * A);
  }

  // check for valid lambdas (filter out intersection points with z < 0 or z > 1)
  if (lambdaCand[0] > 0)
  {
    rayT.rayPos(&rayT, lambdaCand[0], &isectPt);
    if (isectPt.pz < 0 || isectPt.pz > 1)
      lambdaCand[0] = -1;
  }
  if (lambdaCand[1] > 0)
  {
    rayT.rayPos(&rayT, lambdaCand[1], &isectPt);
    if (isectPt.pz < 0 || isectPt.pz > 1)
      lambdaCand[1] = -1;
  }

  // check CYLINDER CAPS intersection //
  // 2 plane intersection, the top cap at z = 1, the bottom cap at z = 0
  lambdaCand[2] = -rayT.p0.pz / rayT.d.pz;      // pz + lambda*dz = 0
  lambdaCand[3] = (1 - rayT.p0.pz) / rayT.d.pz; // pz + lambda*dz = 1

  // check for valid lambdas (filter out intersection points with x^2 + y^2 > 1)
  if (lambdaCand[2] > 0)
  {
    rayT.rayPos(&rayT, lambdaCand[2], &isectPt);
    if (1 < isectPt.px * isectPt.px + isectPt.py * isectPt.py)
      lambdaCand[2] = -1;
  }
  if (lambdaCand[3] > 0)
  {
    rayT.rayPos(&rayT, lambdaCand[3], &isectPt);
    if (1 < isectPt.px * isectPt.px + isectPt.py * isectPt.py)
      lambdaCand[3] = -1;
  }

  // compare intersections to find the smallest lambda //
  *lambda = -1;
  for (int i = 0; i <= 3; i++)
  {
    if ((*lambda < 0 || lambdaCand[i] < *lambda) && lambdaCand[i] > 0)
      *lambda = lambdaCand[i];
  }
  if (*lambda < 0)
    return;

  // step 3: Compute values in world coordinates ////////////////////////////////
  ray->rayPos(ray, *lambda, p);

  struct point3D canNormal;
  if (*lambda == lambdaCand[0] || *lambda == lambdaCand[1])
  { // intersected cylinder wall, normal is xy coordinates of intersection
    rayT.rayPos(&rayT, *lambda, &canNormal);
    canNormal.pz = 0;
    canNormal.pw = 0;

    // set texture coordinates (similar to sphere for a)
    rayT.rayPos(&rayT, *lambda, &isectPt);
    *a = 0.5 + atan2(isectPt.py, isectPt.px) / (2 * PI); // angle along xy-plane (longitude)
    *b = isectPt.pz;                                     // z coordinate
  }
  else if (*lambda == lambdaCand[2])
  { // intersected cylinder bottom cap (z=0), normal points towards negative z
    canNormal.px = 0;
    canNormal.py = 0;
    canNormal.pz = -1;
    canNormal.pw = 0;

    // caps are not considered for texture mapping
    *a = -1;
    *b = -1;
  }
  else if (*lambda == lambdaCand[3])
  { // intersected cylinder top cap (z=1), normal points towards positive z
    canNormal.px = 0;
    canNormal.py = 0;
    canNormal.pz = 1;
    canNormal.pw = 0;

    // caps are not considered for texture mapping
    *a = -1;
    *b = -1;
  }

  // Compute normal
  // adjust normal according to normal map (if the object is normal mapped)
  if (cylinder->normalMapped == 1 && cylinder->normalMap != NULL && *a > 0 && *b > 0)
  {
    struct point3D biTanVec = {.px = 0, .py = 0, .pz = 1, .pw = 0}; // bi-tangent vector in the 'b' texture coordinate direction
    struct point3D *tanVec = cross(&biTanVec, &canNormal);   // tangent vector in the 'a' texture coordinate direction

    normMap(cylinder->normalMap, *a, *b, &canNormal, tanVec, &biTanVec);
    normalize(&canNormal);
  }
  normalTransform(&canNormal, n, cylinder);
}

/**
 * Computes and returns the value of 'lambda' at the intersection
 * between the specified ray and the specified canonical cone.
 *
 * The canonical cone:
 *   radius = 1,
 *   height = 1,
 *   the apex (point) is aligned at the origin
 *   the base is unit circle, parallel with the xy plane at z = 1
 *
 * @param cylinder The cone object.
 * @param ray      The ray object in world coordinate frame.
 * @param lambda   [out] The 'lambda' of the ray at the intersection, set to negative if no intersection.
 * @param p        [out] The intersection point.
 * @param n        [out] The normal at the intersection.
 * @param a        [out] The 'a' texture coordinate (FOR ADVANCED RAY TRACER).
 * @param b        [out] The 'b' texture coordinate (FOR ADVANCED RAY TRACER).
 */
void coneIntersect(struct object3D *cone, struct ray3D *ray, double *lambda, struct point3D *p, struct point3D *n, double *a, double *b)
{
  /*
   * Reference: PACO's lecture notes
   *
   * cone slant: x^2 + y^2   =  z^2    {0 < z < 1}
   * cone base:  x^2 + y^2  <=  1      {z = 0}
   * ray: p + lambda * d    (d is not normalized because of transforms!)
   */

  // Step 1: Get transformed ray (T^-1 * ray)
  struct ray3D rayT;
  initRay(&rayT, &(ray->p0), &(ray->d));
  rayTransform(ray, &rayT, cone);

  // Step 2: find intersection
  // candidate lambdas. index 0,1: slant intersection. index 2: z=-1 base intersection.
  double lambdaCand[4] = {-1, -1, -1};

  // check CONE SLANT intersection //
  // solve (px + lambda * dx)^2 + (py + lambda * dy)^2 - (pz + lambda * dz)^2  =  0
  // kinda like dot product with z component negative.
  double A = rayT.d.px * rayT.d.px + rayT.d.py * rayT.d.py - rayT.d.pz * rayT.d.pz;          // d.d
  double B = 2 * (rayT.p0.px * rayT.d.px + rayT.p0.py * rayT.d.py - rayT.p0.pz * rayT.d.pz); // 2 * d.p
  double C = rayT.p0.px * rayT.p0.px + rayT.p0.py * rayT.p0.py - rayT.p0.pz * rayT.p0.pz;    // p.p

  double discriminant = B * B - 4 * A * C;
  struct point3D isectPt;

  if (discriminant > 0)
  {
    lambdaCand[0] = (-B - sqrt(discriminant)) / (2 * A);
    lambdaCand[1] = (-B + sqrt(discriminant)) / (2 * A);
  }

  // check for valid lambdas (filter out intersection points with z < -1 or z > 0)
  if (lambdaCand[0] > 0)
  {
    rayT.rayPos(&rayT, lambdaCand[0], &isectPt);
    if (isectPt.pz < -1 || isectPt.pz > 0)
      lambdaCand[0] = -1;
  }
  if (lambdaCand[1] > 0)
  {
    rayT.rayPos(&rayT, lambdaCand[1], &isectPt);
    if (isectPt.pz < -1 || isectPt.pz > 0)
      lambdaCand[1] = -1;
  }

  // check CONE BASE intersection //
  // plane intersection at z = -1
  lambdaCand[2] = (-1 - rayT.p0.pz) / rayT.d.pz; // pz + lambda*dz = -1

  // check for valid lambdas (filter out intersection points with x^2 + y^2 > 1)
  if (lambdaCand[2] > 0)
  {
    rayT.rayPos(&rayT, lambdaCand[2], &isectPt);
    if (1 < isectPt.px * isectPt.px + isectPt.py * isectPt.py)
      lambdaCand[2] = -1;
  }

  // compare intersections to find the smallest lambda //
  *lambda = -1;
  for (int i = 0; i <= 2; i++)
  {
    if ((*lambda < 0 || lambdaCand[i] < *lambda) && lambdaCand[i] > 0)
      *lambda = lambdaCand[i];
  }
  if (*lambda < 0)
    return;

  // step 3: Compute values in world coordinates
  ray->rayPos(ray, *lambda, p);

  struct point3D canNormal;
  if (*lambda == lambdaCand[0] || *lambda == lambdaCand[1])
  { // intersected cone slant
    // since our canonical cone has slope 1 (45 deg), rotating by 90 deg is equivalent to flipping along the z axis.
    rayT.rayPos(&rayT, *lambda, &canNormal);
    canNormal.pz = -canNormal.pz;
    canNormal.pw = 0;
    normalize(&canNormal);
  }
  else if (*lambda == lambdaCand[2])
  { // intersected cone base (z=-1), normal points towards negative z
    canNormal.px = 0;
    canNormal.py = 0;
    canNormal.pz = -1;
    canNormal.pw = 0;
  }
  normalTransform(&canNormal, n, cone);

  *a = -1;
  *b = -1;
}

/**
 * Computes and returns the value of 'lambda' at the intersection
 * between the specified ray and the specified canonical cube.
 *
 * The canonical sphere has length 2, and is centered at the origin.
 *
 * @param cube   The cube object.
 * @param ray     The ray object in world coordinate frame.
 * @param lambda  [out] The 'lambda' of the ray at the intersection, set to `-1` if no intersection.
 * @param p       [out] The intersection point.
 * @param n       [out] The normal at the intersection.
 * @param a       [out] The 'a' texture coordinate (FOR ADVANCED RAY TRACER).
 * @param b       [out] The 'b' texture coordinate (FOR ADVANCED RAY TRACER).
 */
void cubeIntersect(struct object3D *cube, struct ray3D *ray, double *lambda, struct point3D *p, struct point3D *n, double *a, double *b)
{
  // Step 1: Get transformed ray (T^-1 * ray)
  struct ray3D ray_transformed;
  initRay(&ray_transformed, &(ray->p0), &(ray->d));
  rayTransform(ray, &ray_transformed, cube);

  // Step 2: Intersect transformed ray with the 6 planes of the cube

  // Check if ray is parallel to any of the the plane
  if ((ray_transformed.d.pz == 0 && (ray_transformed.d.py == 1 || ray_transformed.d.py == -1 || ray_transformed.d.pz == 1 || ray_transformed.d.pz == -1)) || (ray_transformed.d.py == 0 && (ray_transformed.d.px == 1 || ray_transformed.d.px == -1 || ray_transformed.d.pz == 1 || ray_transformed.d.pz == -1)) || (ray_transformed.d.px == 0 && (ray_transformed.d.py == 1 || ray_transformed.d.py == -1 || ray_transformed.d.pz == 1 || ray_transformed.d.pz == -1)))
  {
    *lambda = -1;
    return;
  }

  double lambdaCand[6] = {-1, -1, -1, -1, -1, -1};

  // Solve for lambda
  lambdaCand[0] = (-1 - ray_transformed.p0.pz) / ray_transformed.d.pz; // z=-1
  lambdaCand[1] = (1 - ray_transformed.p0.pz) / ray_transformed.d.pz;  // z=1
  lambdaCand[2] = (-1 - ray_transformed.p0.py) / ray_transformed.d.py; // y=-1
  lambdaCand[3] = (1 - ray_transformed.p0.py) / ray_transformed.d.py;  // y=1
  lambdaCand[4] = (-1 - ray_transformed.p0.px) / ray_transformed.d.px; // x=-1
  lambdaCand[5] = (1 - ray_transformed.p0.px) / ray_transformed.d.px;  // x=1

  double minLambda = -1;
  struct point3D canNormal;

  for (int i = 0; i < 6; i++)
  {
    if (lambdaCand[i] > 0)
    {
      struct point3D p_transformed;
      ray_transformed.rayPos(&ray_transformed, lambdaCand[i], &p_transformed);

      if ((i == 0 || i == 1) && (p_transformed.px > 1 || p_transformed.px < -1 || p_transformed.py > 1 || p_transformed.py < -1))
        continue;
      if ((i == 2 || i == 3) && (p_transformed.px > 1 || p_transformed.px < -1 || p_transformed.pz > 1 || p_transformed.pz < -1))
        continue;
      if ((i == 4 || i == 5) && (p_transformed.py > 1 || p_transformed.py < -1 || p_transformed.pz > 1 || p_transformed.pz < -1))
        continue;

      if (minLambda < 0 || lambdaCand[i] < minLambda)
      {
        minLambda = lambdaCand[i];

        // set normal based on which plane was intersected
        if (i == 0)
        {
          canNormal.px = 0;
          canNormal.py = 0;
          canNormal.pz = -1;
          canNormal.pw = 0;
        }
        else if (i == 1)
        {
          canNormal.px = 0;
          canNormal.py = 0;
          canNormal.pz = 1;
          canNormal.pw = 0;
        }
        else if (i == 2)
        {
          canNormal.px = 0;
          canNormal.py = -1;
          canNormal.pz = 0;
          canNormal.pw = 0;
        }
        else if (i == 3)
        {
          canNormal.px = 0;
          canNormal.py = 1;
          canNormal.pz = 0;
          canNormal.pw = 0;
        }
        else if (i == 4)
        {
          canNormal.px = -1;
          canNormal.py = 0;
          canNormal.pz = 0;
          canNormal.pw = 0;
        }
        else if (i == 5)
        {
          canNormal.px = 1;
          canNormal.py = 0;
          canNormal.pz = 0;
          canNormal.pw = 0;
        }
      }
    }
  }

  if (minLambda < 0)
  {
    *lambda = -1;
    return;
  }

  *lambda = minLambda;

  // Compute intersection point
  ray->rayPos(ray, *lambda, p);

  // Compute normal
  struct point3D normal_transformed;
  normalTransform(&canNormal, &normal_transformed, cube);
  *n = normal_transformed;

  // Ignore a and b for now
  *a = -1;
  *b = -1;
}

/////////////////////////////////////////////////////////////////
// Surface coordinates & random sampling on object surfaces
/////////////////////////////////////////////////////////////////
void planeCoordinates(struct object3D *plane, double a, double b, double *x, double *y, double *z)
{
  // Return in (x,y,z) the coordinates of a point on the plane given by the 2 parameters a,b in [0,1].
  // 'a' controls displacement from the left side of the plane, 'b' controls displacement from the
  // bottom of the plane.

  struct point3D transformedpoint = {.px = -1 + 2 * a, .py = -1 + 2 * b, .pz = 0, .pw = 1};

  // apply transformations
  matVecMult(plane->T, &transformedpoint);

  *x = transformedpoint.px;
  *y = transformedpoint.py;
  *z = transformedpoint.pz;
}

void sphereCoordinates(struct object3D *sphere, double a, double b, double *x, double *y, double *z)
{
  // Return in (x,y,z) the coordinates of a point on the sphere given by the 2 parameters a,b.
  // 'a' in [0, 1] corresponds to the spherical coordinate theta (longitude),
  //                  the angle from the positive x-axis to the point (x,y,z) on the xy-plane.
  // 'b' in [0, 1] corresponds to the spherical coordinate phi (latitude),
  //                      the angle of elevation with respect to the xy-plane, with (0,0,1) being at b=PI/2.

  //  x: \(x=\rho \sin \phi \cos \theta \) y: \(y=\rho \sin \phi \sin \theta \) z: \(z=\rho \cos \phi \) 
  struct point3D transformedPoint = {
      .px = cos(b * PI - PI / 2) * cos(a * 2 * PI),
      .py = cos(b * PI - PI / 2) * sin(a * 2 * PI),
      .pz = sin(b * PI - PI / 2),
      .pw = 1};

  matVecMult(sphere->T, &transformedPoint);

  *x = transformedPoint.px;
  *y = transformedPoint.py;
  *z = transformedPoint.pz;
}

void cylCoordinates(struct object3D *cyl, double a, double b, double *x, double *y, double *z)
{
  // Return in (x,y,z) the coordinates of a point on the plane given by the 2 parameters a,b.
  // 'a' in [0, 1] corresponds to angle theta around the cylinder
  // 'b' in [0, 1] corresponds to height from the bottom

  struct point3D transformedPoint = {
      .px = cos(a * 2 * PI),
      .py = sin(a * 2 * PI),
      .pz = b,
      .pw = 1};

  matVecMult(cyl->T, &transformedPoint);

  *x = transformedPoint.px;
  *y = transformedPoint.py;
  *z = transformedPoint.pz;
}

void coneCoordinates(struct object3D *cyl, double a, double b, double *x, double *y, double *z)
{
  // Return in (x,y,z) the coordinates of a point on the plane given by the 2 parameters a,b.
  // 'a' in [0, 2*PI] corresponds to angle theta around the cylinder
  // 'b' in [0, 1] corresponds to height from the bottom

  /////////////////////////////////
  // TO DO: Complete this function.
  /////////////////////////////////
}

void cubeCoordinates(struct object3D *cube, double a, double b, double *x, double *y, double *z)
{
  // Return in (x,y,z) the coordinates of a point on the cube given by the 2 parameters a,b.

  /////////////////////////////////
  // TO DO: Complete this function.
  /////////////////////////////////
}

void planeSample(struct object3D *plane, double *x, double *y, double *z)
{
  // Returns the 3D coordinates (x,y,z) of a randomly sampled point on the plane
  // Sapling should be uniform, meaning there should be an equal change of gedtting
  // any spot on the plane

  double a = drand48();
  double b = drand48();

  planeCoordinates(plane, a, b, x, y, z);
}

void sphereSample(struct object3D *sphere, double *x, double *y, double *z)
{
  // Returns the 3D coordinates (x,y,z) of a randomly sampled point on the sphere
  // Sampling should be uniform - note that this is tricky for a sphere, do some
  // research and document in your report what method is used to do this, along
  // with a reference to your source.

  /*
   * Rejection sampling:
   * 1. generate point inside a canonical cube (side length 2)
   * 2. reject point if point is not within sphere and try again
   * 3. if point is within the sphere, project it onto the surface of the sphere.
   *
   * 52% chance the point is valid, expected number of iterations: 1.91
   */

  struct point3D randPt;

  randPt.px = drand48() * 2 - 1;
  randPt.py = drand48() * 2 - 1;
  randPt.pz = drand48() * 2 - 1;

  while ((randPt.px * randPt.px + randPt.py * randPt.py + randPt.pz * randPt.pz) > 1)
  {
    randPt.px = drand48() * 2 - 1;
    randPt.py = drand48() * 2 - 1;
    randPt.pz = drand48() * 2 - 1;
  }

  normalize(&randPt);
  matVecMult(sphere->T, &randPt);

  *x = randPt.px;
  *y = randPt.py;
  *z = randPt.pz;
}

void cylSample(struct object3D *cyl, double *x, double *y, double *z)
{
  // Returns the 3D coordinates (x,y,z) of a randomly sampled point on the cylinder
  // Sampling should be uniform over the cylinder.

  double a = drand48();
  double b = drand48();

  cylCoordinates(cyl, a, b, x, y, z);
}

void coneSample(struct object3D *cyl, double *x, double *y, double *z)
{
  // Returns the 3D coordinates (x,y,z) of a randomly sampled point on the cylinder
  // Sampling should be uniform over the cylinder.

  /////////////////////////////////
  // TO DO: Complete this function.
  /////////////////////////////////
}

void cubeSample(struct object3D *cube, double *x, double *y, double *z)
{
  // Returns the 3D coordinates (x,y,z) of a randomly sampled point on the cube

  /////////////////////////////////
  // TO DO: Complete this function.
  /////////////////////////////////
}

/////////////////////////////////
// Texture mapping functions
/////////////////////////////////
void loadTexture(struct object3D *o, const char *filename, int type, struct textureNode **t_list)
{
  // Load a texture or normal map image from file and assign it to the
  // specified object.
  // type:   1  ->  Texture map  (RGB, .ppm)
  //         2  ->  Normal map   (RGB, .ppm)
  //         3  ->  Alpha map    (grayscale, .pgm)
  // Stores loaded images in a linked list to avoid replication
  struct image *im;
  struct textureNode *p;

  if (o != NULL)
  {
    // Check current linked list
    p = *(t_list);
    while (p != NULL)
    {
      if (strcmp(&p->name[0], filename) == 0)
      {
        // Found image already on the list
        if (type == 1)
          o->texImg = p->im;
        else if (type == 2)
        {
          o->normalMap = p->im;
          o->normalMapped = 1;
        }
        else
        {
          o->alphaMap = p->im;
          o->alphaMapped = 1;
        }
        return;
      }
      p = p->next;
    }

    // Load this texture image
    if (type == 1 || type == 2)
      im = readPPMimage(filename);
    else if (type == 3)
      im = readPGMimage(filename);

    // Insert it into the texture list
    if (im != NULL)
    {
      p = (struct textureNode *)calloc(1, sizeof(struct textureNode));
      strcpy(&p->name[0], filename);
      p->type = type;
      p->im = im;
      p->next = NULL;
      // Insert into linked list
      if ((*(t_list)) == NULL)
        *(t_list) = p;
      else
      {
        p->next = (*(t_list))->next;
        (*(t_list))->next = p;
      }
      // Assign to object
      if (type == 1)
        o->texImg = im;
      else if (type == 2)
      {
        o->normalMap = im;
        o->normalMapped = 1;
      }
      else
      {
        o->alphaMap = im;
        o->alphaMapped = 1;
      }
    }

  } // end if (o != NULL)
}

void texMap(struct image *img, double a, double b, double *R, double *G, double *B)
{
  /*
   Function to determine the colour of a textured object at
   the normalized texture coordinates (a,b).

   a and b are texture coordinates in [0 1].
   img is a pointer to the image structure holding the texture for
    a given object.

   The colour is returned in R, G, B. Uses bi-linear interpolation
   to determine texture colour.
  */

  if (a < 0 || a > 1 || b < 0 || b > 1)
  { // (a,b) texture coordinates are out of bounds, don't update colour values
    return;
  }

  // scale to get image pixel coordinates (pxlX, pxlY) (double)
  double pxlX, pxlY;
  pxlX = a * (img->sx - 1);
  pxlY = b * (img->sy - 1);

  // linear interpolation for
  struct colourRGB colour;
  interpolatePixel(img, pxlX, pxlY, &colour);

  *(R) = colour.R;
  *(G) = colour.G;
  *(B) = colour.B;
}

/**
 * Helper function to modify the input normal according to the normal map from `img`.
 * The normal, tangent, and bi-tangent vectors are the normalized basis vectors of the
 * tangent space at the object intersection.
 * @param img the normal map image file.
 * @param a,b the horizontal and vertical texture coordinate, respectively.
 * @param normVec [out] the normal computed from intersection test. Will be modified to contain the normal according to normal map.
 * @param tanVec the tangent vector at the intersection point, aligned with the `a` texture coordinate axis.
 * @param biTanVec the bi-tangent vector at the intersection point, aligned with the `b` texture coordinate axis.
 */
void normMap(struct image *img, double a, double b, struct point3D *normVec, struct point3D *tanVec, struct point3D *biTanVec)
{
  if (a < 0 || a > 1 || b < 0 || b > 1)
  { // (a,b) map coordinates are out of bounds, don't update the normal
    return;
  }

  // construct matrix to convert from tangent space to world coordinates
  /*
   * The transformation matrix (fill the homogeneous part with zeros):
   * [     |        |        |     ]
   * [  tanVec  biTanVec  normVec  ]
   * [     |        |        |     ]
   */

  // Set up the tangent space to World space transformation matrix.
  // Mind the indexing convention [row][col]
  double tan2W[4][4];
  memset(&tan2W[0][0], 0, 16 * sizeof(double));

  tan2W[0][0] = tanVec->px;
  tan2W[1][0] = tanVec->py;
  tan2W[2][0] = tanVec->pz;

  tan2W[0][1] = biTanVec->px;
  tan2W[1][1] = biTanVec->py;
  tan2W[2][1] = biTanVec->pz;

  tan2W[0][2] = normVec->px;
  tan2W[1][2] = normVec->py;
  tan2W[2][2] = normVec->pz;

  // Transform the mapped normal vector and store it in `normVec`
  double pxlX, pxlY;
  pxlX = a * (img->sx - 1);
  pxlY = b * (img->sy - 1);

  struct colourRGB normMapVec = {.R = 0, .G = 0, .B = 1}; // just in case, set the default to the unmapped normal
  interpolatePixel(img, pxlX, pxlY, &normMapVec);               // get mapped normal from the normal map.

  // put the tangent space normal into `normVec` and transform to world coordinates
  normVec->px = normMapVec.R * 2 - 1; // transform [0,1] to [-1,1]
  normVec->py = normMapVec.G * 2 - 1; // transform [0,1] to [-1,1]
  normVec->pz = normMapVec.B;         // assume the normal direction is [0,1]
  normVec->pw = 0;

  matVecMult(tan2W, normVec);
}

void alphaMap(struct image *img, double a, double b, double *alpha)
{
  // Just like texture map but returns the alpha value at a,b,
  // notice that alpha maps are single layer grayscale images, hence
  // the separate function.

  if (a < 0 || a > 1 || b < 0 || b > 1)
  { // (a,b) texture coordinates are out of bounds, assume opaque
    *(alpha) = 1;
    return;
  }

  // scale to get image pixel coordinates (pxlX, pxlY) (double)
  double pxlX, pxlY;
  pxlX = a * (img->sx - 1);
  pxlY = b * (img->sy - 1);

  // Get the integer parts of x and y
  int x0 = (int)pxlX;
  int y0 = (int)pxlY;

  // Clamp to image boundaries to avoid accessing out-of-bounds pixels
  int x1 = x0 + 1 < img->sx ? x0 + 1 : x0;
  int y1 = y0 + 1 < img->sy ? y0 + 1 : y0;

  // Calculate the fractional parts of x and y
  float x_frac = pxlX - x0;
  float y_frac = pxlY - y0;

  // Fetch the four neighboring pixels
  double p00 = ((double *)(img->rgbdata)) [(y0 * img->sx + x0) + 0];  // Top-left
  double p10 = ((double *)(img->rgbdata)) [(y0 * img->sx + x1) + 0];  // Top-right
  double p01 = ((double *)(img->rgbdata)) [(y1 * img->sx + x0) + 0];  // Bottom-left
  double p11 = ((double *)(img->rgbdata)) [(y1 * img->sx + x1) + 0];  // Bottom-right

  // Perform bi-linear interpolation for each color channel
  *alpha = (
          p00 * (1 - x_frac) * (1 - y_frac) +
          p10 * x_frac * (1 - y_frac) +
          p01 * (1 - x_frac) * y_frac +
          p11 * x_frac * y_frac
  );

  printf("[debug] alpha %f\n", *alpha);
}

/////////////////////////////
// Light sources
/////////////////////////////
void insertPLS(struct pointLS *l, struct pointLS **list)
{
  if (l == NULL)
    return;
  // Inserts a light source into the list of light sources
  if (*(list) == NULL)
  {
    *(list) = l;
    (*(list))->next = NULL;
  }
  else
  {
    l->next = (*(list))->next;
    (*(list))->next = l;
  }
}

void addAreaLight(struct object3D *newObj(double ra, double rd, double rs, double rg, double r, double g, double b, double alpha, double R_index, double shiny),
                  double sx, double sy, double sz, double nx, double ny, double nz,
                  double tx, double ty, double tz, int N,
                  double r, double g, double b, struct object3D **o_list, struct pointLS **l_list)
{
  /*
    This function sets up and inserts an object area light source
    scaled by (sx, sy, sz)
    orientation given by the normal vector (nx, ny, nz)
    centered at (tx, ty, tz)
    consisting of (N) point light sources (uniformly sampled)
    and with colour/intensity (r,g,b)

    Note that the light source must be visible as a uniformly colored rectangle which
    casts no shadows. If you require a lightsource to shade another, you must
    make it into a proper solid box with a back and sides of non-light-emitting
    material
  */

  // NOTE: The best way to implement area light sources is to random sample from the
  //       light source's object surface within rtShade(). This is a bit more tricky
  //       but reduces artifacts significantly. If you do that, then there is no need
  //       to insert a series of point lightsources in this function.

  // Step 1: get obj with a size of (sx, sy) and is centered at (tx, ty, tz) with normal vector (nx, ny, nz)
  struct object3D *obj = newObj(1, 0, 0, 0, r, g, b, 1, 1, 1);
  obj->isLightSource = 1;

  // scale the obj by (sx, sy, sz)
  Scale(obj, sx, sy, sz);

  // rotate the obj to have normal vector (nx, ny, nz)

  // get normalized normal vector
  struct point3D n = {.px = nx, .py = ny, .pz = nz, .pw = 0};
  normalize(&n);

  double xRot = atan2(n.py, n.pz);
  double yRot = atan2(n.px, sqrt(n.py * n.py + n.pz * n.pz));

  RotateY(obj, yRot);
  RotateX(obj, -xRot);

  // translate the obj to (tx, ty, tz)
  Translate(obj, tx, ty, tz);

  // set T inverse matrix
  invert(&obj->T[0][0], &obj->Tinv[0][0]);

  // Step 2: insert the obj into the object list
  insertObject(obj, o_list);

  // Step 3: insert N point light sources into the light source list
  for (int i = 0; i < N; i++)
  {
    double x, y, z;

    obj->randomPoint(obj, &x, &y, &z);

    struct point3D pos = {.px = x, .py = y, .pz = z, .pw = 1};
    struct pointLS *pls = newPLS(&pos, r / N, g / N, b / N);

    insertPLS(pls, l_list);
  }
}

///////////////////////////////////
// Geometric transformation section
///////////////////////////////////

/**
 * Computes the inverse of transformation matrix T.
 * the result is returned in Tinv.
 * @param T    The matrix to compute the inverse of.
 * @param Tinv [out] place to output the inverse.
 */
void invert(double *T, double *Tinv)
{
  double *U, *s, *V, *rv1;
  int singFlag, i;

  // Invert the affine transform
  U = NULL;
  s = NULL;
  V = NULL;
  rv1 = NULL;
  singFlag = 0;

  SVD(T, 4, 4, &U, &s, &V, &rv1);
  if (U == NULL || s == NULL || V == NULL)
  {
    fprintf(stderr, "Error: Matrix not invertible for this object, returning identity\n");
    memcpy(Tinv, eye4x4, 16 * sizeof(double));
    return;
  }

  // Check for singular matrices...
  for (i = 0; i < 4; i++)
    if (*(s + i) < 1e-9)
      singFlag = 1;
  if (singFlag)
  {
    fprintf(stderr, "Error: Transformation matrix is singular, returning identity\n");
    memcpy(Tinv, eye4x4, 16 * sizeof(double));
    return;
  }

  // Compute and store inverse matrix
  InvertMatrix(U, s, V, 4, Tinv);

  free(U);
  free(s);
  free(V);
}

void RotateXMat(double T[4][4], double theta)
{
  // Multiply the current object transformation matrix T in object o
  // by a matrix that rotates the object theta *RADIANS* around the
  // X axis.

  double R[4][4];
  memset(&R[0][0], 0, 16 * sizeof(double));

  R[0][0] = 1.0;
  R[1][1] = cos(theta);
  R[1][2] = -sin(theta);
  R[2][1] = sin(theta);
  R[2][2] = cos(theta);
  R[3][3] = 1.0;

  matMult(R, T);
}

void RotateX(struct object3D *o, double theta)
{
  // Multiply the current object transformation matrix T in object o
  // by a matrix that rotates the object theta *RADIANS* around the
  // X axis.

  double R[4][4];
  memset(&R[0][0], 0, 16 * sizeof(double));

  R[0][0] = 1.0;
  R[1][1] = cos(theta);
  R[1][2] = -sin(theta);
  R[2][1] = sin(theta);
  R[2][2] = cos(theta);
  R[3][3] = 1.0;

  matMult(R, o->T);
}

void RotateYMat(double T[4][4], double theta)
{
  // Multiply the current object transformation matrix T in object o
  // by a matrix that rotates the object theta *RADIANS* around the
  // Y axis.

  double R[4][4];
  memset(&R[0][0], 0, 16 * sizeof(double));

  R[0][0] = cos(theta);
  R[0][2] = sin(theta);
  R[1][1] = 1.0;
  R[2][0] = -sin(theta);
  R[2][2] = cos(theta);
  R[3][3] = 1.0;

  matMult(R, T);
}

void RotateY(struct object3D *o, double theta)
{
  // Multiply the current object transformation matrix T in object o
  // by a matrix that rotates the object theta *RADIANS* around the
  // Y axis.

  double R[4][4];
  memset(&R[0][0], 0, 16 * sizeof(double));

  R[0][0] = cos(theta);
  R[0][2] = sin(theta);
  R[1][1] = 1.0;
  R[2][0] = -sin(theta);
  R[2][2] = cos(theta);
  R[3][3] = 1.0;

  matMult(R, o->T);
}

void RotateZMat(double T[4][4], double theta)
{
  // Multiply the current object transformation matrix T in object o
  // by a matrix that rotates the object theta *RADIANS* around the
  // Z axis.

  double R[4][4];
  memset(&R[0][0], 0, 16 * sizeof(double));

  R[0][0] = cos(theta);
  R[0][1] = -sin(theta);
  R[1][0] = sin(theta);
  R[1][1] = cos(theta);
  R[2][2] = 1.0;
  R[3][3] = 1.0;

  matMult(R, T);
}

void RotateZ(struct object3D *o, double theta)
{
  // Multiply the current object transformation matrix T in object o
  // by a matrix that rotates the object theta *RADIANS* around the
  // Z axis.

  double R[4][4];
  memset(&R[0][0], 0, 16 * sizeof(double));

  R[0][0] = cos(theta);
  R[0][1] = -sin(theta);
  R[1][0] = sin(theta);
  R[1][1] = cos(theta);
  R[2][2] = 1.0;
  R[3][3] = 1.0;

  matMult(R, o->T);
}

void TranslateMat(double T[4][4], double tx, double ty, double tz)
{
  // Multiply the current object transformation matrix T in object o
  // by a matrix that translates the object by the specified amounts.

  double tr[4][4];
  memset(&tr[0][0], 0, 16 * sizeof(double));

  tr[0][0] = 1.0;
  tr[1][1] = 1.0;
  tr[2][2] = 1.0;
  tr[0][3] = tx;
  tr[1][3] = ty;
  tr[2][3] = tz;
  tr[3][3] = 1.0;

  matMult(tr, T);
}

void Translate(struct object3D *o, double tx, double ty, double tz)
{
  // Multiply the current object transformation matrix T in object o
  // by a matrix that translates the object by the specified amounts.

  double tr[4][4];
  memset(&tr[0][0], 0, 16 * sizeof(double));

  tr[0][0] = 1.0;
  tr[1][1] = 1.0;
  tr[2][2] = 1.0;
  tr[0][3] = tx;
  tr[1][3] = ty;
  tr[2][3] = tz;
  tr[3][3] = 1.0;

  matMult(tr, o->T);
}

void ScaleMat(double T[4][4], double sx, double sy, double sz)
{
  // Multiply the current object transformation matrix T in object o
  // by a matrix that scales the object as indicated.

  double S[4][4];
  memset(&S[0][0], 0, 16 * sizeof(double));

  S[0][0] = sx;
  S[1][1] = sy;
  S[2][2] = sz;
  S[3][3] = 1.0;

  matMult(S, T);
}

void Scale(struct object3D *o, double sx, double sy, double sz)
{
  // Multiply the current object transformation matrix T in object o
  // by a matrix that scales the object as indicated.

  double S[4][4];
  memset(&S[0][0], 0, 16 * sizeof(double));

  S[0][0] = sx;
  S[1][1] = sy;
  S[2][2] = sz;
  S[3][3] = 1.0;

  matMult(S, o->T);
}

void printmatrix(double mat[4][4])
{
  fprintf(stderr, "Matrix contains:\n");
  fprintf(stderr, "%f %f %f %f\n", mat[0][0], mat[0][1], mat[0][2], mat[0][3]);
  fprintf(stderr, "%f %f %f %f\n", mat[1][0], mat[1][1], mat[1][2], mat[1][3]);
  fprintf(stderr, "%f %f %f %f\n", mat[2][0], mat[2][1], mat[2][2], mat[2][3]);
  fprintf(stderr, "%f %f %f %f\n", mat[3][0], mat[3][1], mat[3][2], mat[3][3]);
}

/////////////////////////////////////////
// Camera and view setup
/////////////////////////////////////////
struct view *setupView(struct point3D *e, struct point3D *g, struct point3D *up, double f, double wl, double wt, double wsize)
{
  /*
    This function sets up the camera axes and viewing direction as discussed in the
    lecture notes.
    e - Camera center
    g - Gaze direction
    up - Up vector
    fov - Fild of view in degrees
    f - focal length
  */
  struct view *c;
  struct point3D *u, *v;

  u = v = NULL;

  // Allocate space for the camera structure
  c = (struct view *)calloc(1, sizeof(struct view));
  if (c == NULL)
  {
    fprintf(stderr, "Out of memory setting up camera model!\n");
    return (NULL);
  }

  // Set up camera center and axes
  c->e.px = e->px; // Copy camera center location, note we must make sure
  c->e.py = e->py; // the camera center provided to this function has pw=1
  c->e.pz = e->pz;
  c->e.pw = 1;

  // Set up w vector (camera's Z axis). w=-g/||g||
  c->w.px = -g->px;
  c->w.py = -g->py;
  c->w.pz = -g->pz;
  c->w.pw = 1;
  normalize(&c->w);

  // Set up the horizontal direction, which must be perpenticular to w and up
  u = cross(&c->w, up);
  normalize(u);
  c->u.px = u->px;
  c->u.py = u->py;
  c->u.pz = u->pz;
  c->u.pw = 1;

  // Set up the remaining direction, v=(u x w)  - Mind the signs
  v = cross(&c->u, &c->w);
  normalize(v);
  c->v.px = v->px;
  c->v.py = v->py;
  c->v.pz = v->pz;
  c->v.pw = 1;

  // Copy focal length and window size parameters
  c->f = f;
  c->wl = wl;
  c->wt = wt;
  c->wsize = wsize;

  // Set up coordinate conversion matrices
  // Camera2World matrix (M_cw in the notes)
  // Mind the indexing convention [row][col]
  c->C2W[0][0] = c->u.px;
  c->C2W[1][0] = c->u.py;
  c->C2W[2][0] = c->u.pz;
  c->C2W[3][0] = 0;

  c->C2W[0][1] = c->v.px;
  c->C2W[1][1] = c->v.py;
  c->C2W[2][1] = c->v.pz;
  c->C2W[3][1] = 0;

  c->C2W[0][2] = c->w.px;
  c->C2W[1][2] = c->w.py;
  c->C2W[2][2] = c->w.pz;
  c->C2W[3][2] = 0;

  c->C2W[0][3] = c->e.px;
  c->C2W[1][3] = c->e.py;
  c->C2W[2][3] = c->e.pz;
  c->C2W[3][3] = 1;

  // World2Camera matrix (M_wc in the notes)
  // Mind the indexing convention [row][col]
  c->W2C[0][0] = c->u.px;
  c->W2C[1][0] = c->v.px;
  c->W2C[2][0] = c->w.px;
  c->W2C[3][0] = 0;

  c->W2C[0][1] = c->u.py;
  c->W2C[1][1] = c->v.py;
  c->W2C[2][1] = c->w.py;
  c->W2C[3][1] = 0;

  c->W2C[0][2] = c->u.pz;
  c->W2C[1][2] = c->v.pz;
  c->W2C[2][2] = c->w.pz;
  c->W2C[3][2] = 0;

  c->W2C[0][3] = -dot(&c->u, &c->e);
  c->W2C[1][3] = -dot(&c->v, &c->e);
  c->W2C[2][3] = -dot(&c->w, &c->e);
  c->W2C[3][3] = 1;

  free(u);
  free(v);
  return (c);
}

/////////////////////////////////////////
// Image I/O section
/////////////////////////////////////////
struct image *readPPMimage(const char *filename)
{
  // Reads an image from a .ppm file. A .ppm file is a very simple image representation
  // format with a text header followed by the binary RGB data at 24bits per pixel.
  // The header has the following form:
  //
  // P6
  // # One or more comment lines preceded by '#'
  // 340 200
  // 255
  //
  // The first line 'P6' is the .ppm format identifier, this is followed by one or more
  // lines with comments, typically used to inidicate which program generated the
  // .ppm file.
  // After the comments, a line with two integer values specifies the image resolution
  // as number of pixels in x and number of pixels in y.
  // The final line of the header stores the maximum value for pixels in the image,
  // usually 255.
  // After this last header line, binary data stores the RGB values for each pixel
  // in row-major order. Each pixel requires 3 bytes ordered R, G, and B.
  //
  // NOTE: Windows file handling is rather crotchetty. You may have to change the
  //       way this file is accessed if the images are being corrupted on read
  //       on Windows.
  //
  // readPPMdata converts the image colour information to floating point. This is so that
  // the texture mapping function doesn't have to do the conversion every time
  // it is asked to return the colour at a specific location.
  //

  FILE *f;
  struct image *im;
  char line[1024];
  int sizx, sizy;
  int i;
  unsigned char *tmp;
  double *fRGB;
  int tmpi;
  char *tmpc;

  im = (struct image *)calloc(1, sizeof(struct image));
  if (im != NULL)
  {
    im->rgbdata = NULL;
    f = fopen(filename, "rb+");
    if (f == NULL)
    {
      fprintf(stderr, "Unable to open file %s for reading, please check name and path\n", filename);
      free(im);
      return (NULL);
    }
    tmpc = fgets(&line[0], 1000, f);
    if (strcmp(&line[0], "P6\n") != 0)
    {
      fprintf(stderr, "Wrong file format, not a .ppm file or header end-of-line characters missing\n");
      free(im);
      fclose(f);
      return (NULL);
    }
    fprintf(stderr, "%s\n", line);
    // Skip over comments
    tmpc = fgets(&line[0], 511, f);
    while (line[0] == '#')
    {
      fprintf(stderr, "%s", line);
      tmpc = fgets(&line[0], 511, f);
    }
    sscanf(&line[0], "%d %d\n", &sizx, &sizy); // Read file size
    fprintf(stderr, "nx=%d, ny=%d\n\n", sizx, sizy);
    im->sx = sizx;
    im->sy = sizy;

    tmpc = fgets(&line[0], 9, f); // Read the remaining header line
    fprintf(stderr, "%s\n", line);
    tmp = (unsigned char *)calloc(sizx * sizy * 3, sizeof(unsigned char));
    fRGB = (double *)calloc(sizx * sizy * 3, sizeof(double));
    if (tmp == NULL || fRGB == NULL)
    {
      fprintf(stderr, "Out of memory allocating space for image\n");
      free(im);
      fclose(f);
      return (NULL);
    }

    tmpi = fread(tmp, sizx * sizy * 3 * sizeof(unsigned char), 1, f);
    fclose(f);

    // Conversion to floating point
    for (i = 0; i < sizx * sizy * 3; i++)
      *(fRGB + i) = ((double)*(tmp + i)) / 255.0;
    free(tmp);
    im->rgbdata = (void *)fRGB;

    return (im);
  }

  fprintf(stderr, "Unable to allocate memory for image structure\n");
  return (NULL);
}

struct image *readPGMimage(const char *filename)
{
  // Just like readPPMimage() except it is used to load grayscale alpha maps. In
  // alpha maps, a value of 255 corresponds to alpha=1 (fully opaque) and 0
  // correspondst to alpha=0 (fully transparent).
  // A .pgm header of the following form is expected:
  //
  // P5
  // # One or more comment lines preceded by '#'
  // 340 200
  // 255
  //
  // readPGMdata converts the image grayscale data to double floating point in [0,1].

  FILE *f;
  struct image *im;
  char line[1024];
  int sizx, sizy;
  int i;
  unsigned char *tmp;
  double *fRGB;
  int tmpi;
  char *tmpc;

  im = (struct image *)calloc(1, sizeof(struct image));
  if (im != NULL)
  {
    im->rgbdata = NULL;
    f = fopen(filename, "rb+");
    if (f == NULL)
    {
      fprintf(stderr, "Unable to open file %s for reading, please check name and path\n", filename);
      free(im);
      return (NULL);
    }
    tmpc = fgets(&line[0], 1000, f);
    if (strcmp(&line[0], "P5\n") != 0)
    {
      fprintf(stderr, "Wrong file format, not a .pgm file or header end-of-line characters missing\n");
      free(im);
      fclose(f);
      return (NULL);
    }
    // Skip over comments
    tmpc = fgets(&line[0], 511, f);
    while (line[0] == '#')
      tmpc = fgets(&line[0], 511, f);
    sscanf(&line[0], "%d %d\n", &sizx, &sizy); // Read file size
    im->sx = sizx;
    im->sy = sizy;

    tmpc = fgets(&line[0], 9, f); // Read the remaining header line
    tmp = (unsigned char *)calloc(sizx * sizy, sizeof(unsigned char));
    fRGB = (double *)calloc(sizx * sizy, sizeof(double));
    if (tmp == NULL || fRGB == NULL)
    {
      fprintf(stderr, "Out of memory allocating space for image\n");
      free(im);
      fclose(f);
      return (NULL);
    }

    tmpi = fread(tmp, sizx * sizy * sizeof(unsigned char), 1, f);
    fclose(f);

    // Conversion to double floating point
    for (i = 0; i < sizx * sizy; i++)
      *(fRGB + i) = ((double)*(tmp + i)) / 255.0;
    free(tmp);
    im->rgbdata = (void *)fRGB;

    return (im);
  }

  fprintf(stderr, "Unable to allocate memory for image structure\n");
  return (NULL);
}

struct image *newImage(int size_x, int size_y)
{
  // Allocates and returns a new image with all zeros. Assumes 24 bit per pixel,
  // unsigned char array.
  struct image *im;

  im = (struct image *)calloc(1, sizeof(struct image));
  if (im != NULL)
  {
    im->rgbdata = NULL;
    im->sx = size_x;
    im->sy = size_y;
    im->rgbdata = (void *)calloc(size_x * size_y * 3, sizeof(unsigned char));
    if (im->rgbdata != NULL)
      return (im);
  }
  fprintf(stderr, "Unable to allocate memory for new image\n");
  return (NULL);
}

void imageOutput(struct image *im, const char *filename)
{
  // Writes out a .ppm file from the image data contained in 'im'.
  // Note that Windows typically doesn't know how to open .ppm
  // images. Use Gimp or any other seious image processing
  // software to display .ppm images.
  // Also, note that because of Windows file format management,
  // you may have to modify this file to get image output on
  // Windows machines to work properly.
  //
  // Assumes a 24 bit per pixel image stored as unsigned chars
  //

  FILE *f;

  if (im != NULL)
    if (im->rgbdata != NULL)
    {
      f = fopen(filename, "wb+");
      if (f == NULL)
      {
        fprintf(stderr, "Unable to open file %s for output! No image written\n", filename);
        return;
      }
      fprintf(f, "P6\n");
      fprintf(f, "# Output from RayTracer.c\n");
      fprintf(f, "%d %d\n", im->sx, im->sy);
      fprintf(f, "255\n");
      fwrite((unsigned char *)im->rgbdata, im->sx * im->sy * 3 * sizeof(unsigned char), 1, f);
      fclose(f);
      return;
    }
  fprintf(stderr, "imageOutput(): Specified image is empty. Nothing output\n");
}

void deleteImage(struct image *im)
{
  // De-allocates memory reserved for the image stored in 'im'
  if (im != NULL)
  {
    if (im->rgbdata != NULL)
      free(im->rgbdata);
    free(im);
  }
}

void cleanup(struct object3D *o_list, struct pointLS *l_list, struct textureNode *t_list)
{
  // De-allocates memory reserved for the object list and the point light source
  // list. Note that *YOU* must de-allocate any memory reserved for images
  // rendered by the raytracer.
  struct object3D *p, *q;
  struct pointLS *r, *s;
  struct textureNode *t, *u;

  p = o_list; // De-allocate all memory from objects in the list
  while (p != NULL)
  {
    q = p->next;
    if (p->photonMap != NULL) // If object is photon mapped, free photon map memory
    {
      if (p->photonMap->rgbdata != NULL)
        free(p->photonMap->rgbdata);
      free(p->photonMap);
    }
    free(p);
    p = q;
  }

  r = l_list; // Delete light source list
  while (r != NULL)
  {
    s = r->next;
    free(r);
    r = s;
  }

  t = t_list; // Delete texture Images
  while (t != NULL)
  {
    u = t->next;
    if (t->im->rgbdata != NULL)
      free(t->im->rgbdata);
    free(t->im);
    free(t);
    t = u;
  }
}

// used to generate the cool scene using buildscene-procedural.c
void generateCoolScene(double ra, double rd, double rs, double rg, double alpha, double r_index, double shinyness,
                       double x, double y, double z,
                       double translateX, double translateY, double translateZ, double rot,
                       int isRGB, double numIters, object3D **object_list)
{
  double rgb[6][3] =
      {
          {1, .25, .25},
          {1, .75, .25},
          {1, 1, .25},
          {.25, 1, .25},
          {.25, .25, 1},
          {.5, .25, .75},
      };

  if (isRGB == 0)
  {
    memset(rgb, (double)0.2, sizeof(double) * 6 * 3);
  };

  for (int i = 0; i < numIters; i++)
  {
    struct object3D *o = newCyl(ra, rd, rs, rg, rgb[i % 6][0], rgb[i % 6][1], rgb[i % 6][2], alpha, r_index, shinyness);

    Scale(o, 2, .75, .75);
    RotateZ(o, PI / 4);

    RotateX(o, rot * i);

    Translate(o, x, y, z);
    Translate(o, translateX * i, translateY * i, translateZ * i);
    invert(&o->T[0][0], &o->Tinv[0][0]);
    insertObject(o, object_list);
  }
}

void drawTree(double size, int depth, double T_mat[4][4], struct albedosPhong branchPh, struct colourRGB barkCol, struct colourRGB jointCol, double shiny, object3D *object_list)
{
  branchOne(size, depth, T_mat, branchPh, barkCol, jointCol, shiny, object_list);
}

/**
 * Draw one branch from the current node
 */
void branchOne(double size, int depth, double T_mat[4][4], struct albedosPhong branchPh, struct colourRGB barkCol, struct colourRGB jointCol, double shiny, object3D *object_list)
{
  double jointRatio = 1.1;  // Radius of the joint sphere, if the cylinder has radius 1
  double cylRatio = 5;      // Height of the cylinder, if the cylinder has radius 1
  double recurseRatio = 0.8;

  struct object3D *o;

  // draw trunk (cylinder)
  o = newCyl(branchPh.ra, branchPh.rd, branchPh.rs, branchPh.rg, barkCol.R, barkCol.G, barkCol.B, 1, 1, shiny);
  RotateX(o, -PI/2);
  Scale(o, size, size*cylRatio, size);
//  Translate(o, 0, size*cylRatio, 0);
  matMult(T_mat, o->T);    // apply hierarchical transform
  invert(&o->T[0][0], &o->Tinv[0][0]);
  insertObject(o, &object_list);

  if (depth > 0)
  {
    // draw joint (sphere)
    o = newSphere(branchPh.ra, branchPh.rd, branchPh.rs, branchPh.rg, jointCol.R, jointCol.G, jointCol.B, 1, 1, shiny);
    Scale(o, size*jointRatio, size*jointRatio, size*jointRatio);
    Translate(o, 0, size*cylRatio, 0);
    matMult(T_mat, o->T);    // apply hierarchical transform
    invert(&o->T[0][0], &o->Tinv[0][0]);
    insertObject(o, &object_list);

    // recursively call drawTree(...)
    double tempMat[4][4];
    double thetaX = 0, thetaY = 0;

    for (int i = 0; i < 3; i++) {
      thetaX = drand48() * PI/3;
      thetaY = drand48() * PI*2;

      memcpy(&tempMat[0][0], &eye4x4[0][0], 16 * sizeof(double));
      RotateZMat(tempMat, thetaX);
      RotateYMat(tempMat, thetaY);
      TranslateMat(tempMat, 0, size*cylRatio, 0);
      matMult(T_mat, tempMat);
      drawTree(size*recurseRatio, depth - 1, tempMat, branchPh, barkCol, jointCol, shiny, object_list);
    }
  }
  else {
    // draw leaf ////////////////////////////////////////////////////////////////////////////
    double leafSize = drand48() + 0.3;

    o = newSphere(branchPh.ra, branchPh.rd, branchPh.rs, 0.1, 0.15, 0.45, 0.25, 1, 1, shiny);
    Scale(o, leafSize, leafSize, leafSize);
    Translate(o, 0, size*cylRatio, 0);
    matMult(T_mat, o->T);    // apply hierarchical transform
    invert(&o->T[0][0], &o->Tinv[0][0]);
    insertObject(o, &object_list);
  }
}