#include "utils.h"

struct object3D *obj;
struct pointLS *l;
struct point3D p;

// tunable parameters

struct colourRGB white = {.R = .9, .G = .9, .B = .9};
struct colourRGB blue  = {.R = .1, .G = .1, .B = .9};
struct colourRGB red   = {.R = .9, .G = .1, .B = .1};

struct albedosPhong phong = {.ra = .1, .rd = .9, .rs = .2, .rg = .1};
double wallShinny = 2;

// CAMERA ////////////////////////////////////////////////////////////////////////////////
struct point3D e = {.px = 0, .py = 2, .pz = -4, .pw = 1};    // Camera center
struct point3D g = {    // Camera gaze vector
        .px = 0 - e.px,
        .py = 0.5 - e.py,
        .pz = 0 - e.pz,
        .pw = 1
};
struct point3D up = {.px = 0, .py = 1, .pz = 0, .pw = 1};    // Camera up vector

cam = setupView(&e, &g, &up, -2.5, -1, 1, 2);

// LIGHT ////////////////////////////////////////////////////////////////////////////////

struct point3D s = {.px = 1, .py = 1, .pz = 1, .pw = 0}; // scale
struct point3D t = {.px = 3, .py = 5, .pz = -3, .pw = 1}; // translate
struct point3D n = {
        .px = 0 - t.px,
        .py = 0 - t.px,
        .pz = 0 - t.pz,
        .pw = 0}; // normal (towards the origin)

addAreaLight(newPlane, s.px, s.py, s.pz, n.px, n.py, n.pz, t.px, t.py, t.pz,
             100, .95, .95, .95, &object_list, &light_list);

// GLOBE ////////////////////////////////////////////////////////////////////////////////
struct object3D *globe = newSphere(phong.ra, phong.rd, phong.rs, phong.rg,  white.R, white.G, white.B ,  1, 1, 4);
loadTexture(globe, "assets/2k_earth_daymap.ppm", 1, &texture_list);
loadTexture(globe, "assets/earth_normal_map.ppm", 2, &texture_list);
loadTexture(globe, "assets/earth_alpha_map.pgm", 3, &texture_list);
RotateX(globe, PI/2);
invert(&globe->T[0][0], &globe->Tinv[0][0]);
insertObject(globe, &object_list);

// GROUND PLANE ////////////////////////////////////////////////////////////////////////////////
obj = newPlane(phong.ra, phong.rd, .1, phong.rg, white.R, white.G, white.B, 1, 1, wallShinny);
//loadTexture(obj, "assets/testTexture.ppm", 1, &texture_list);
loadTexture(obj, "assets/wood-grain-texture.ppm", 1, &texture_list);
Scale(obj, 8, 8, 1);
RotateX(obj, PI/2);
RotateY(obj, PI/4);
Translate(obj, 0, -1, 0);
invert(&obj->T[0][0], &obj->Tinv[0][0]);
insertObject(obj, &object_list);

// HANGING STARS ////////////////////////////////////////////////////////////////////////////////
// stars laid out in a semicircle `star_distance` away from the origin in the direction of the gaze vector

int num_stars = 4;
double ceilHeight = 10;

//double X[4] = {
//        dist*(-sin(-.5)),
//        dist*(-sin(-.25)),
//        dist*(-sin(.25)),
//        dist*(-sin(5))
//};

double Y[4] = {
        0.4,
        0.5,
        0.4,
        0.4
};
//double Z[4] = {
//        dist*(cos(-.5)),
//        dist*(cos(-.25)),
//        dist*(cos(.25)),
//        dist*(cos(.5))
//};

double angles[4] = {
        -.5,
        -.25,
        .25,
        .5
};

double dist[4] = {
        7,
        6,
        4.5,
        5
};


double radius[4] = {
        .5,
        .7,
        .4,
        .7
};

for (int i = 0; i < num_stars; i++) {
  drawStar(dist[i]*(-sin(angles[i])),
           Y[i],
           dist[i]*(cos(angles[i])),
           radius[i],
           ceilHeight,
           &texture_list, &object_list);
}
