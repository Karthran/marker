#pragma once

#define MARKER_H
//#include "stdafx.h"

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/highgui/highgui.hpp"

#include <cctype>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <iostream>

class Inv_Moments
{
public:
    //! the default constructor
    Inv_Moments();

    //! spatial moments
    double  m00, m10, m01, m20, m11, m02, m30, m21, m12, m03;
    //! central moments
    double  mu20, mu11, mu02, mu30, mu21, mu12, mu03;
    //! central normalized moments
    double  nu20, nu11, nu02, nu30, nu21, nu12, nu03;
    //!Hu invariant
    double hu0, hu1, hu2, hu3, hu4, hu5, hu6;
    //! Full invariant
    double M0, M1, M2, M3, M4, M5, M6;
    void clear();
};
Inv_Moments::Inv_Moments()
{
    clear();
}
void Inv_Moments::clear() {
    m00 = m10 = m01 = m20 = m11 = m02 = m30 = m21 = m12 = m03 =
        mu20 = mu11 = mu02 = mu30 = mu21 = mu12 = mu03 =
        nu20 = nu11 = nu02 = nu30 = nu21 = nu12 = nu03 =
        hu0 = hu1 = hu2 = hu3 = hu4 = hu5 = hu6 =
        M0 = M1 = M2 = M3 = M4 = M5 = M6 = 0;
}