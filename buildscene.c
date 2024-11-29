void buildScene(void)
{
  double eye4x4[4][4] = {{1.0, 0.0, 0.0, 0.0},
                         {0.0, 1.0, 0.0, 0.0},
                         {0.0, 0.0, 1.0, 0.0},
                         {0.0, 0.0, 0.0, 1.0}};

  struct object3D *o;
  struct point3D p;

  struct albedosPhong phong = {.ra = .3, .rd = .9, .rs = .05, .rg = 0};

  struct colourRGB grassGreen = {.R = 0.4, .G = 0.7, .B = 0.2};
  struct colourRGB barkCol = {.R = 0.7, .G = 0.5, .B = 0.1};
  struct colourRGB jointCol = {.R = 0.8, .G = 0.6, .B = 0.1};
  struct colourRGB white = {.R = 0.9, .G = 0.9, .B = 0.9};

  double groundShinny = 1;
  double groundSize = 100;


  // Scene /////////////////////////////////////////////////////////////////
  // ground
  o = newPlane(phong.ra, phong.rd, phong.rs, 0.1,
               grassGreen.R, grassGreen.G, grassGreen.B,
               1, 1, groundShinny);
  Scale(o, groundSize, groundSize, 1);
  RotateX(o, PI/2);
//  Translate(o, 0, 0, 0);
  invert(&o->T[0][0], &o->Tinv[0][0]);
  insertObject(o, &object_list);

  // draw tree
  struct albedosPhong branchPh = {.ra = .3, .rd = .9, .rs = .05, .rg = 0};
  double shiny = 3;

  double T_mat[4][4];
  memcpy(&T_mat[0][0], &eye4x4[0][0], 16 * sizeof(double));
//  RotateYMat(T_mat, PI/2);

  drawTree(0.5, 6, T_mat, branchPh, barkCol, jointCol, 3, object_list);


  // Light /////////////////////////////////////////////////////////////////
  p.px = 20;
  p.py = 80;
  p.pz = -20;
  p.pw = 1;
  struct pointLS *light = newPLS(&p, white.R, white.G, white.B);
  insertPLS(light, &light_list);

  // Camera ////////////////////////////////////////////////////////////////
  struct point3D e = {.px = 0, .py = 6, .pz = -20, .pw = 1};    // Camera center
  struct point3D g = {    // Camera gaze vector
          .px = 0 - e.px,
          .py = 5 - e.py,
          .pz = 0 - e.pz,
          .pw = 1
  };
  struct point3D up = {.px = 0, .py = 1, .pz = 0, .pw = 1};    // Camera up vector

  cam = setupView(&e, &g, &up, -2.5, -1, 1, 2);
}