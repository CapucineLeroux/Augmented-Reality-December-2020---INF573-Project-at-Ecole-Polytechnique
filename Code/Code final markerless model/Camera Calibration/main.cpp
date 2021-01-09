#define CERES_FOUND true
#define OPENCV_TRAITS_ENABLE_DEPRECATED
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include "opencv2/xfeatures2d.hpp"
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core.hpp>
#include <opencv2/sfm.hpp>
#include <opencv2/viz.hpp>
#include <iostream>
#include <fstream>
#include <opencv2/sfm/reconstruct.hpp>

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;

//finds the interior corners of a chessboard on an image
vector<Point2f> find_corners(string path){

    Size patternsize(7,7); //interior number of corners
    Mat gray = imread(path,IMREAD_GRAYSCALE); //source image

    vector<Point2f> corners; //this will be filled by the detected corners

    bool patternfound = findChessboardCorners(gray, patternsize, corners,
            CALIB_CB_ADAPTIVE_THRESH + CALIB_CB_NORMALIZE_IMAGE
            + CALIB_CB_FAST_CHECK);

    if(patternfound){
        cornerSubPix(gray, corners, Size(11, 11), Size(-1, -1),
        TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 30, 0.1));
    }

    return corners;

};

int main()
{
    //build a matrix with the 3D coordinates of the chessboard interior corners
    int CHECKERBOARD[2]{7,7};
    vector<Point3f> corners_3d;
    for(int i{0}; i<CHECKERBOARD[1]; i++){
        for(int j{0}; j<CHECKERBOARD[0]; j++){
            corners_3d.push_back(Point3f(j,i,0));
        }
      }

    vector<vector<Point2f>> im_pts; //stores 2D coordinates of the chessboard interior corners
    vector<Point2f> corners; //stocks the detected corners on the image
    vector<vector<Point3f>> obj_pts; //stores 3D coordinates of the chessboard interior corners

    //goes through the collection of images of the chessboard and detect corners
    for (int i=1 ; i<=16 ; i++){
        corners = find_corners("../chessboards_test/chessboard"+to_string(i)+".jpg");
        im_pts.push_back(corners);
        obj_pts.push_back(corners_3d);
        cout<<"image "<<i<<" : done"<<endl;
    }

    //builds the camera intrinsic matrix
    Mat cameraMatrix,distCoeffs,R,T;
    Mat gray = imread("../chessboards_test/chessboard1.jpg",IMREAD_GRAYSCALE);
    calibrateCamera(obj_pts, im_pts, Size(gray.rows,gray.cols), cameraMatrix, distCoeffs, R, T);
    cout << "cameraMatrix : " << cameraMatrix << endl;
    cout << "distCoeffs : " << distCoeffs << endl;

}
