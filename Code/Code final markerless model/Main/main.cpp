#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <iostream>
#include <opencv2/sfm.hpp>
#include <chrono>
#include <opencv2/aruco.hpp>

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;

Mat I1; //will stock a frame of the video
vector<Point2d> imagePoints; // stocks the 4 base 2D points of the cube
vector<Mat> frames; //will stock all the frames with the added 3D cube to print the result at the end



//recollects the pose of the cube in the first frame
void onMouse(int event, int x, int y, int foo, void* p)
{
        if (event != EVENT_LBUTTONDOWN)
                return;
    Point2d m(x,y);
    circle(I1, m, 2, Scalar(0, 255, 0), 2);
    imshow("Initialisation", I1);
    imagePoints.push_back(m);
}



//multiplies an euclidian 2d point with 3x3 matrix H
Point2d multiply(Mat H, Point2d p)
{
    Mat_<double> src(3,1);

    src(0,0)=p.x;
    src(1,0)=p.y;
    src(2,0)=1.0;

    Mat_<double> dst = H*src;

    return Point2d(dst(0,0)/dst(2,0),dst(1,0)/dst(2,0));
}



//project the cube in the image and draw it
void drawcube(Mat I, vector<Point2d> imgPts, vector<Point3d> objPts, Mat& K, Mat& distCoeffs, Mat& rvec, Mat& tvec, Mat& rmatrix, Mat& P, Mat cube){

    Mat imOut = I.clone(); //will be the frame with the cube drawn on it

    //solves the projection matrices

    solvePnP(objPts, imgPts, K, distCoeffs, rvec, tvec, false, SOLVEPNP_IPPE_SQUARE);

    aruco::drawAxis(imOut, K, distCoeffs, rvec, tvec, 1.0);

    //transform R and T into matrices
    Rodrigues(rvec, rmatrix);

    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            P.at<double>(i,j) = rmatrix.at<double>(i,j);
        }
        P.at<double>(i,3) = tvec.at<double>(i,0);
    }

    //Projects the cube
    Mat cube_T;

    transpose(cube, cube_T);

    Mat transformed = K * P * cube_T;

    Mat transformed_euclidean;
    sfm::homogeneousToEuclidean(transformed, transformed_euclidean);

    // Draw the edges of the cube

    Point start, end;
    int thickness = 2;

    // Plane y = 0

    start.x = transformed_euclidean.at<double>(0,0);
    start.y = transformed_euclidean.at<double>(1,0);
    end.x = transformed_euclidean.at<double>(0,1);
    end.y = transformed_euclidean.at<double>(1,1);

    line(imOut, start, end, Scalar(0,255,0), thickness);

    start.x = transformed_euclidean.at<double>(0,1);
    start.y = transformed_euclidean.at<double>(1,1);
    end.x = transformed_euclidean.at<double>(0,2);
    end.y = transformed_euclidean.at<double>(1,2);

    line(imOut, start, end, Scalar(0,255,0),thickness);

    start.x = transformed_euclidean.at<double>(0,2);
    start.y = transformed_euclidean.at<double>(1,2);
    end.x = transformed_euclidean.at<double>(0,3);
    end.y = transformed_euclidean.at<double>(1,3);

    line(imOut, start, end, Scalar(0,255,0),thickness);

    start.x = transformed_euclidean.at<double>(0,3);
    start.y = transformed_euclidean.at<double>(1,3);
    end.x = transformed_euclidean.at<double>(0,0);
    end.y = transformed_euclidean.at<double>(1,0);

    line(imOut, start, end, Scalar(0,255,0),thickness);


    // Plane y = 1

    start.x = transformed_euclidean.at<double>(0,4);
    start.y = transformed_euclidean.at<double>(1,4);
    end.x = transformed_euclidean.at<double>(0,5);
    end.y = transformed_euclidean.at<double>(1,5);

    line(imOut, start, end, Scalar(255,255,0),thickness);

    start.x = transformed_euclidean.at<double>(0,5);
    start.y = transformed_euclidean.at<double>(1,5);
    end.x = transformed_euclidean.at<double>(0,6);
    end.y = transformed_euclidean.at<double>(1,6);

    line(imOut, start, end, Scalar(255,255,0),thickness);

    start.x = transformed_euclidean.at<double>(0,6);
    start.y = transformed_euclidean.at<double>(1,6);
    end.x = transformed_euclidean.at<double>(0,7);
    end.y = transformed_euclidean.at<double>(1,7);

    line(imOut, start, end, Scalar(255,255,0),thickness);

    start.x = transformed_euclidean.at<double>(0,7);
    start.y = transformed_euclidean.at<double>(1,7);
    end.x = transformed_euclidean.at<double>(0,4);
    end.y = transformed_euclidean.at<double>(1,4);

    line(imOut, start, end, Scalar(255,255,0),thickness);

    // Vertical ones

    start.x = transformed_euclidean.at<double>(0,0);
    start.y = transformed_euclidean.at<double>(1,0);
    end.x = transformed_euclidean.at<double>(0,4);
    end.y = transformed_euclidean.at<double>(1,4);

    line(imOut, start, end, Scalar(0,255,255),thickness);

    start.x = transformed_euclidean.at<double>(0,1);
    start.y = transformed_euclidean.at<double>(1,1);
    end.x = transformed_euclidean.at<double>(0,5);
    end.y = transformed_euclidean.at<double>(1,5);

    line(imOut, start, end, Scalar(0,255,255),thickness);


    start.x = transformed_euclidean.at<double>(0,2);
    start.y = transformed_euclidean.at<double>(1,2);
    end.x = transformed_euclidean.at<double>(0,6);
    end.y = transformed_euclidean.at<double>(1,6);

    line(imOut, start, end, Scalar(0,255,255),thickness);


    start.x = transformed_euclidean.at<double>(0,3);
    start.y = transformed_euclidean.at<double>(1,3);
    end.x = transformed_euclidean.at<double>(0,7);
    end.y = transformed_euclidean.at<double>(1,7);

    line(imOut, start, end, Scalar(0,255,255),thickness);

    imshow("Calculating...", imOut);
    frames.push_back(imOut);
    //waitKey(0);
}



//first slower version to update the 4 reference points of the cube by applying an homography
void update(Mat I_prec, Mat I, vector<Point2d>& imgPts){

    Ptr<AKAZE> D = AKAZE::create();
    vector<KeyPoint> m1, m2;
    Mat descriptors1, descriptors2;
    D->detectAndCompute( I_prec, noArray(), m1, descriptors1 );
    D->detectAndCompute( I, noArray(), m2, descriptors2 );

    BFMatcher M(NORM_HAMMING);
    vector<vector<DMatch>> matches;
    //catches the 2 closest neighbours of descriptors1 in descriptors2
    M.knnMatch(descriptors1,descriptors2,matches,2);

    //compares the 2 closest neighbours, if there is a big difference of distance, then it's a good match
    float threshold = 0.8f;
    vector<DMatch> matched;
    vector<KeyPoint> keypoints1, keypoints2;

    DMatch first_neighbour;
    float distance1;
    float distance2;
    int new_i;
    //only keeps good matches in matched and keypoints1/2
    for(size_t i = 0; i < matches.size(); i++) {

        first_neighbour = matches[i][0];
        distance1 = matches[i][0].distance;
        distance2 = matches[i][1].distance;

        if(distance1 < threshold*distance2) {
            new_i = static_cast<int>(keypoints1.size());
            keypoints1.push_back(m1[first_neighbour.queryIdx]);
            keypoints2.push_back(m2[first_neighbour.trainIdx]);
            matched.push_back(DMatch(new_i, new_i, 0));
        }
    }

    vector<Point2f> points1, points2;
    for(size_t i=0 ; i<matched.size() ; i++){
        points1.push_back( keypoints1[matched[i].queryIdx].pt );
        points2.push_back( keypoints2[matched[i].trainIdx].pt );
    }

    //calculates the homography and the corresponding mask to draw the inliers matches
    vector<char> mask;
    Mat H;
    H = findHomography(points1, points2, RANSAC, 3, mask);

    Point2d p;
    Mat I_points = I.clone();
    for (int i=0 ; i<imgPts.size() ; i++){
        p = imgPts[i];
        p = multiply(H,p);
        imgPts[i] = p;
        circle(I_points, p, 2, Scalar(0, 255, 0), 2);
    }

    //imshow("Calculating...", I_points);

}



//second quicker version to update the 4 reference points of the cube by applying an homography
void quick_update(Mat I_prec, Mat I, vector<Point2d>& imgPts){

    //feature detection and description

    vector<KeyPoint> fp1; //feature points
    Mat descriptors1; //descriptors
    Ptr<FastFeatureDetector> Detection1 = FastFeatureDetector::create(); //use FAST method to detect feature points
    Detection1->detect(I_prec, fp1); //store the detected feature points of the frame in fp
    Ptr<SURF> Description1 = SURF::create(); //use SURF method to compute the description of the feature points
    Description1->compute(I_prec, fp1, descriptors1); //store the description in descriptors

    //same thing for the following frame
    vector<KeyPoint> fp2;
    Mat descriptors2;
    Ptr<FastFeatureDetector> Detection2 = FastFeatureDetector::create();
    Detection2->detect(I, fp2);
    Ptr<SURF> Description2 = SURF::create();
    Description2->compute(I, fp2, descriptors2);

    //matching

    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED); //uses the matcher FLANN with the norm L2
    std::vector< std::vector<DMatch> > knn_matches;
    matcher->knnMatch( descriptors1, descriptors2, knn_matches, 2 ); //finds the two nearest neighbours and stores it in knn_matches

    //filters matches using the Lowe's ratio test
    const float ratio_thresh = 0.7f;
    vector<DMatch> good_matches;

    for (size_t i = 0; i < knn_matches.size(); i++)
    {
        if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance)
        {
            good_matches.push_back(knn_matches[i][0]);
        }
    }

    vector<Point2f> points1, points2;
    for(size_t i=0 ; i<good_matches.size() ; i++){
        points1.push_back( fp1[good_matches[i].queryIdx].pt );
        points2.push_back( fp2[good_matches[i].trainIdx].pt );
    }

    //calculates the homography and the corresponding mask to draw the inliers matches
    vector<char> mask;
    Mat H;
    H = findHomography(points1, points2, RANSAC, 3, mask);

    //update the 2D coordinates of the 4 anchors points of the cube
    Point2d p;
    Mat I_points = I.clone();
    for (int i=0 ; i<imgPts.size() ; i++){
        p = imgPts[i];
        p = multiply(H,p);
        imgPts[i] = p;
        circle(I_points, p, 2, Scalar(0, 255, 0), 2);
    }

    //imshow("Calculating...", I_points);

}



//recollects the initialisation, calculates and draws the cube on the video
void print_calculating_video(string video_path){

    //recollects the initialisation
    VideoCapture cap(video_path);
    cap >> I1;
    imshow("Initialisation",I1);
    setMouseCallback("Initialisation", onMouse);
    waitKey(0);

    //anchors points 3D coordinates
    vector<Point3d> objectPoints = {Point3d(0,1,0), Point3d(1,1,0), Point3d(1,0,0), Point3d(0,0,0)};

    //intrinsic camera matrix, values were found with the Camera Calibration script
    Mat K = Mat::eye(3, 3, CV_64F);
    K.at<double>(0,0) = 1765;
    K.at<double>(1,1) = 1767;
    K.at<double>(0,2) = I1.cols/2.;
    K.at<double>(1,2) = I1.rows/2.;

    //3D coordinates of the cube
    Mat cube = Mat(8, 4, CV_64F);
    cube.at<double>(0,0) = 0; cube.at<double>(0,1) = 0; cube.at<double>(0,2) = 0; cube.at<double>(0,3) = 1;
    cube.at<double>(1,0) = 1; cube.at<double>(1,1) = 0; cube.at<double>(1,2) = 0; cube.at<double>(1,3) = 1;
    cube.at<double>(2,0) = 1; cube.at<double>(2,1) = 0; cube.at<double>(2,2) = 1; cube.at<double>(2,3) = 1;
    cube.at<double>(3,0) = 0; cube.at<double>(3,1) = 0; cube.at<double>(3,2) = 1; cube.at<double>(3,3) = 1;
    cube.at<double>(4,0) = 0; cube.at<double>(4,1) = 1; cube.at<double>(4,2) = 0; cube.at<double>(4,3) = 1;
    cube.at<double>(5,0) = 1; cube.at<double>(5,1) = 1; cube.at<double>(5,2) = 0; cube.at<double>(5,3) = 1;
    cube.at<double>(6,0) = 1; cube.at<double>(6,1) = 1; cube.at<double>(6,2) = 1; cube.at<double>(6,3) = 1;
    cube.at<double>(7,0) = 0; cube.at<double>(7,1) = 1; cube.at<double>(7,2) = 1; cube.at<double>(7,3) = 1;

    // For solvePnP later
    Mat distCoeffs = Mat::zeros(Size(1, 5), CV_64FC1);
    distCoeffs.at<double>(0,0) = 0.46;
    distCoeffs.at<double>(0,1) = -2.1;
    distCoeffs.at<double>(0,2) = 0.02;
    distCoeffs.at<double>(0,3) = 0.02;
    distCoeffs.at<double>(0,4) = 4.25;
    Mat rvec;
    Mat tvec;
    Mat rmatrix;
    Mat P = Mat(3, 4, CV_64F);

    drawcube(I1, imagePoints, objectPoints, K, distCoeffs, rvec, tvec, rmatrix, P, cube);

    Mat I2;
    int counter = 0;
    while(1){

        // Capture frame-by-frame
        cap >> I2;

        // If the frame is empty, break immediately
        if (I2.empty())
          break;

        if (counter%2 == 0){

            //double t1 = chrono::duration_cast<chrono::duration<double>>(chrono::system_clock::now().time_since_epoch()).count();
            quick_update(I1,I2,imagePoints);
            drawcube(I2, imagePoints, objectPoints, K, distCoeffs, rvec, tvec, rmatrix, P, cube);
            //double t2 = chrono::duration_cast<chrono::duration<double>>(chrono::system_clock::now().time_since_epoch()).count();
            //cout<<"duration : "<<t2-t1<<endl;

            I1 = I2.clone();
        }

        counter += 1;

        // Press  ESC on keyboard to exit
        char c=(char)waitKey(25);
        if(c==27)
          break;

    }

    // When everything done, release the video capture object
    cap.release();
    // Closes all the frames
    destroyAllWindows();
    cout<<"nb of frames "<<counter<<endl;
}



//prints the final result at the end
void print_final_video(string video_path){

    VideoCapture cap(video_path);
    Mat frame;
    cap >> frame;
    int counter = 0;
    int frame_counter = 0;
    imshow("frame",frames[frame_counter]);
    while(1){

        // Capture frame-by-frame
        cap >> frame;
        // If the frame is empty, break immediately
        if (frame.empty())
          break;

        if (counter%2 == 0){
            imshow("Result",frames[frame_counter]);
            frame_counter += 1;
        }

        counter += 1;
        // Press  ESC on keyboard to exit
        char c=(char)waitKey(25);
        if(c==27)
          break;

    }

    // When everything done, release the video capture object
    cap.release();
    // Closes all the frames
    destroyAllWindows();

};



//takes the path of the video in argument
int main(int argc, char *argv[])
{
    print_calculating_video(argv[1]);
    print_final_video(argv[1]);

    return 0;
}

