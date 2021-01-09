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
#include <opencv2/aruco.hpp>

using namespace std;
using namespace cv;


int main()
{
    // Create the marker
    Mat marker;

    // Load the predefined dictionary
    Ptr<cv::aruco::Dictionary> dictionary = aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250);

    // Generate the marker
    aruco::drawMarker(dictionary, 33, 200, marker, 1);

    imshow("marker", marker);
    imwrite("marker.jpg", marker);

    // Initialize the detector parameters using default values
    Ptr<cv::aruco::DetectorParameters> parameters = cv::aruco::DetectorParameters::create();

    // Declare the vectors that would contain the detected marker corners and the rejected marker candidates
    vector<vector<Point2f>> markerCorners, rejectedCandidates;

    // The ids of the detected markers are stored in a vector
    vector<int> markerIds;

    vector<Point3d> objectPoints = {Point3d(0,1,0), Point3d(1,1,0), Point3d(1,0,0), Point3d(0,0,0)};

    // *** Choosing the source of the video stream *** //
    // Make source = 0 for the webcam and source != 0 for a prerecorded video

    int source = 0;
    cv::VideoCapture stream;

    if (source == 0) {

        // Open the first webcam plugged in the computer
        stream = cv::VideoCapture(0);

    } else {

        // Select the source path
        String path = "";
        stream = cv::VideoCapture(path);
    }

    if (!stream.isOpened()) {
        std::cerr << "ERROR: Could not open video source" << std::endl;
        return 1;
    }

    // Create a window to display the images from the video stream
    cv::namedWindow("Original", CV_WINDOW_AUTOSIZE);

    // This will contain the current frame of the video stream
    cv::Mat frame;
    stream >> frame;

    // For calibration
    int count_marker_detections = 0;
    vector<vector<Point2f>> points2d;
    Mat K = Mat::eye(3, 3, CV_64F);
    K.at<double>(0,0) = 50;
    K.at<double>(1,1) = 50;
    K.at<double>(0,2) = frame.cols/2.;
    K.at<double>(1,2) = frame.rows/2.;
    Mat distCoeffs;


    Mat cube = Mat(8, 4, CV_64F);
    cube.at<double>(0,0) = 0; cube.at<double>(0,1) = 0; cube.at<double>(0,2) = 0; cube.at<double>(0,3) = 1;
    cube.at<double>(1,0) = 1; cube.at<double>(1,1) = 0; cube.at<double>(1,2) = 0; cube.at<double>(1,3) = 1;
    cube.at<double>(2,0) = 1; cube.at<double>(2,1) = 0; cube.at<double>(2,2) = 1; cube.at<double>(2,3) = 1;
    cube.at<double>(3,0) = 0; cube.at<double>(3,1) = 0; cube.at<double>(3,2) = 1; cube.at<double>(3,3) = 1;
    cube.at<double>(4,0) = 0; cube.at<double>(4,1) = 1; cube.at<double>(4,2) = 0; cube.at<double>(4,3) = 1;
    cube.at<double>(5,0) = 1; cube.at<double>(5,1) = 1; cube.at<double>(5,2) = 0; cube.at<double>(5,3) = 1;
    cube.at<double>(6,0) = 1; cube.at<double>(6,1) = 1; cube.at<double>(6,2) = 1; cube.at<double>(6,3) = 1;
    cube.at<double>(7,0) = 0; cube.at<double>(7,1) = 1; cube.at<double>(7,2) = 1; cube.at<double>(7,3) = 1;


    // Display the frame until you press a key
    while (true) {

        // Capture the next frame
        stream >> frame;

        if(frame.empty()) break; // End of video stream

        // Show the image on the window
        cv::imshow("Original", frame);

        // Wait (10ms) for a key to be pressed
        if (waitKey(10) == 27 || getWindowProperty("Original", WND_PROP_AUTOSIZE) < 0)
            break;

        // Detect the markers in the image
        detectMarkers(frame, dictionary, markerCorners, markerIds, parameters, rejectedCandidates);
        vector<vector<Point3f>> objPoints;
        vector<Point3f> points = {objectPoints[0], objectPoints[1], objectPoints[2], objectPoints[3]};
        objPoints.push_back(points);

        if (markerCorners.size() != 0) {

            Ptr<aruco::Board> board = aruco::Board::create(objPoints, dictionary, markerIds);
            aruco::refineDetectedMarkers(frame, board, markerCorners, markerIds, rejectedCandidates);

            if (count_marker_detections < 10) {

                // Accumulate correspondences to the camera calibration

                vector<Point2f> pts = {markerCorners[0][0], markerCorners[0][1], markerCorners[0][2], markerCorners[0][3]};
                points2d.push_back(pts);
                count_marker_detections++;

            } else if (count_marker_detections == 10) {

                // Do the camera calibration

                vector<int> counter(10, 1);
                aruco::calibrateCameraAruco(points2d, markerIds, counter, board, frame.size(), K, distCoeffs);
                count_marker_detections = 50;

            } else {

                // Calculating the camera rotation and translation

                Mat imOut = frame.clone();

                vector<Point2d> imagePoints = {markerCorners[0][0], markerCorners[0][1], markerCorners[0][2], markerCorners[0][3]};
                Mat rvec;
                Mat tvec;
                Mat rmatrix;

                solvePnP(objectPoints, imagePoints, K, distCoeffs, rvec, tvec, false, SOLVEPNP_IPPE_SQUARE);
                aruco::drawAxis(imOut, K, distCoeffs, rvec, tvec, 1.0);
                Rodrigues(rvec, rmatrix);

                // Constructing the projection matrix P

                Mat P = Mat(3, 4, CV_64F);

                for (int i = 0; i < 3; i++) {
                    for (int j = 0; j < 3; j++) {
                        P.at<double>(i,j) = rmatrix.at<double>(i,j);
                    }
                    P.at<double>(i,3) = tvec.at<double>(i,0);
                }

                // Applying the projection to the cube

                Mat cube_T;
                transpose(cube, cube_T);
                Mat transformed = K * P * cube_T;
                Mat transformed_euclidean;
                sfm::homogeneousToEuclidean(transformed, transformed_euclidean);

                // Draw the edges of the cube

                Point start, end;

                // Plane y = 0

                start.x = transformed_euclidean.at<double>(0,0);
                start.y = transformed_euclidean.at<double>(1,0);
                end.x = transformed_euclidean.at<double>(0,1);
                end.y = transformed_euclidean.at<double>(1,1);

                line(imOut, start, end, Scalar(0,255,0), 2);

                start.x = transformed_euclidean.at<double>(0,1);
                start.y = transformed_euclidean.at<double>(1,1);
                end.x = transformed_euclidean.at<double>(0,2);
                end.y = transformed_euclidean.at<double>(1,2);

                line(imOut, start, end, Scalar(0,255,0), 2);

                start.x = transformed_euclidean.at<double>(0,2);
                start.y = transformed_euclidean.at<double>(1,2);
                end.x = transformed_euclidean.at<double>(0,3);
                end.y = transformed_euclidean.at<double>(1,3);

                line(imOut, start, end, Scalar(0,255,0), 2);

                start.x = transformed_euclidean.at<double>(0,3);
                start.y = transformed_euclidean.at<double>(1,3);
                end.x = transformed_euclidean.at<double>(0,0);
                end.y = transformed_euclidean.at<double>(1,0);

                line(imOut, start, end, Scalar(0,255,0), 2);


                // Plane y = 1

                start.x = transformed_euclidean.at<double>(0,4);
                start.y = transformed_euclidean.at<double>(1,4);
                end.x = transformed_euclidean.at<double>(0,5);
                end.y = transformed_euclidean.at<double>(1,5);

                line(imOut, start, end, Scalar(255,255,0), 2);

                start.x = transformed_euclidean.at<double>(0,5);
                start.y = transformed_euclidean.at<double>(1,5);
                end.x = transformed_euclidean.at<double>(0,6);
                end.y = transformed_euclidean.at<double>(1,6);

                line(imOut, start, end, Scalar(255,255,0), 2);

                start.x = transformed_euclidean.at<double>(0,6);
                start.y = transformed_euclidean.at<double>(1,6);
                end.x = transformed_euclidean.at<double>(0,7);
                end.y = transformed_euclidean.at<double>(1,7);

                line(imOut, start, end, Scalar(255,255,0), 2);

                start.x = transformed_euclidean.at<double>(0,7);
                start.y = transformed_euclidean.at<double>(1,7);
                end.x = transformed_euclidean.at<double>(0,4);
                end.y = transformed_euclidean.at<double>(1,4);

                line(imOut, start, end, Scalar(255,255,0), 2);

                // Vertical ones

                start.x = transformed_euclidean.at<double>(0,0);
                start.y = transformed_euclidean.at<double>(1,0);
                end.x = transformed_euclidean.at<double>(0,4);
                end.y = transformed_euclidean.at<double>(1,4);

                line(imOut, start, end, Scalar(0,255,255), 2);

                start.x = transformed_euclidean.at<double>(0,1);
                start.y = transformed_euclidean.at<double>(1,1);
                end.x = transformed_euclidean.at<double>(0,5);
                end.y = transformed_euclidean.at<double>(1,5);

                line(imOut, start, end, Scalar(0,255,255), 2);


                start.x = transformed_euclidean.at<double>(0,2);
                start.y = transformed_euclidean.at<double>(1,2);
                end.x = transformed_euclidean.at<double>(0,6);
                end.y = transformed_euclidean.at<double>(1,6);

                line(imOut, start, end, Scalar(0,255,255), 2);


                start.x = transformed_euclidean.at<double>(0,3);
                start.y = transformed_euclidean.at<double>(1,3);
                end.x = transformed_euclidean.at<double>(0,7);
                end.y = transformed_euclidean.at<double>(1,7);

                line(imOut, start, end, Scalar(0,255,255), 2);


                imshow("try", imOut);
            }
        }
    }

	return 0;
}

