#include "common.h"
using namespace cv;
using namespace std;
namespace songyuncen_subpixel {
// https://github.com/songyuncen/EdgesSubPix/blob/master/EdgesSubPix.cpp

double getAmplitude(Mat &dx, Mat &dy, int i, int j)
{
    Point2d mag(dx.at<short>(i, j), dy.at<short>(i, j));
    return norm(mag);
}

void getMagNeighbourhood(Mat &dx, Mat &dy, Point &p, int w, int h, vector<double> &mag)
{
    int top = p.y - 1 >= 0 ? p.y - 1 : p.y;
    int down = p.y + 1 < h ? p.y + 1 : p.y;
    int left = p.x - 1 >= 0 ? p.x - 1 : p.x;
    int right = p.x + 1 < w ? p.x + 1 : p.x;

    mag[0] = getAmplitude(dx, dy, top, left);
    mag[1] = getAmplitude(dx, dy, top, p.x);
    mag[2] = getAmplitude(dx, dy, top, right);
    mag[3] = getAmplitude(dx, dy, p.y, left);
    mag[4] = getAmplitude(dx, dy, p.y, p.x);
    mag[5] = getAmplitude(dx, dy, p.y, right);
    mag[6] = getAmplitude(dx, dy, down, left);
    mag[7] = getAmplitude(dx, dy, down, p.x);
    mag[8] = getAmplitude(dx, dy, down, right);
}

void get2ndFacetModelIn3x3(vector<double> &mag, vector<double> &a)
{
    a[0] = (-mag[0] + 2.0 * mag[1] - mag[2] + 2.0 * mag[3] + 5.0 * mag[4] + 2.0 * mag[5] - mag[6] + 2.0 * mag[7] - mag[8]) / 9.0;
    a[1] = (-mag[0] + mag[2] - mag[3] + mag[5] - mag[6] + mag[8]) / 6.0;
    a[2] = (mag[6] + mag[7] + mag[8] - mag[0] - mag[1] - mag[2]) / 6.0;
    a[3] = (mag[0] - 2.0 * mag[1] + mag[2] + mag[3] - 2.0 * mag[4] + mag[5] + mag[6] - 2.0 * mag[7] + mag[8]) / 6.0;
    a[4] = (-mag[0] + mag[2] + mag[6] - mag[8]) / 4.0;
    a[5] = (mag[0] + mag[1] + mag[2] - 2.0 * (mag[3] + mag[4] + mag[5]) + mag[6] + mag[7] + mag[8]) / 6.0;
}
/*
   Compute the eigenvalues and eigenvectors of the Hessian matrix given by
   dfdrr, dfdrc, and dfdcc, and sort them in descending order according to
   their absolute values.
*/
void eigenvals(vector<double> &a, double eigval[2], double eigvec[2][2])
{
    // derivatives
    // fx = a[1], fy = a[2]
    // fxy = a[4]
    // fxx = 2 * a[3]
    // fyy = 2 * a[5]
    double dfdrc = a[4];
    double dfdcc = a[3] * 2.0;
    double dfdrr = a[5] * 2.0;
    double theta, t, c, s, e1, e2, n1, n2; /* , phi; */

    /* Compute the eigenvalues and eigenvectors of the Hessian matrix. */
    if (dfdrc != 0.0) {
        theta = 0.5*(dfdcc - dfdrr) / dfdrc;
        t = 1.0 / (fabs(theta) + sqrt(theta*theta + 1.0));
        if (theta < 0.0) t = -t;
        c = 1.0 / sqrt(t*t + 1.0);
        s = t*c;
        e1 = dfdrr - t*dfdrc;
        e2 = dfdcc + t*dfdrc;
    }
    else {
        c = 1.0;
        s = 0.0;
        e1 = dfdrr;
        e2 = dfdcc;
    }
    n1 = c;
    n2 = -s;

    /* If the absolute value of an eigenvalue is larger than the other, put that
    eigenvalue into first position.  If both are of equal absolute value, put
    the negative one first. */
    if (fabs(e1) > fabs(e2)) {
        eigval[0] = e1;
        eigval[1] = e2;
        eigvec[0][0] = n1;
        eigvec[0][1] = n2;
        eigvec[1][0] = -n2;
        eigvec[1][1] = n1;
    }
    else if (fabs(e1) < fabs(e2)) {
        eigval[0] = e2;
        eigval[1] = e1;
        eigvec[0][0] = -n2;
        eigvec[0][1] = n1;
        eigvec[1][0] = n1;
        eigvec[1][1] = n2;
    }
    else {
        if (e1 < e2) {
            eigval[0] = e1;
            eigval[1] = e2;
            eigvec[0][0] = n1;
            eigvec[0][1] = n2;
            eigvec[1][0] = -n2;
            eigvec[1][1] = n1;
        }
        else {
            eigval[0] = e2;
            eigval[1] = e1;
            eigvec[0][0] = -n2;
            eigvec[0][1] = n1;
            eigvec[1][0] = n1;
            eigvec[1][1] = n2;
        }
    }
}

// end https://github.com/songyuncen/EdgesSubPix/blob/master/EdgesSubPix.cpp
}
