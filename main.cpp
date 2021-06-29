#include <QCoreApplication>
#include <stdio.h>
#include <iostream>
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/nonfree/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include "opencv2/legacy/legacy.hpp"





using namespace cv;
using namespace std;


int main(int argc, char *argv[])
{
    QCoreApplication a(argc, argv);

    Mat img1 = imread("testc1.png", CV_LOAD_IMAGE_COLOR);
    Mat img1o;
    Mat img2= imread("testc2.png", CV_LOAD_IMAGE_COLOR);
    Mat img2o;

    if (!img1.data || !img2.data)
                   return 0;



    // vector of keypoints
    vector <KeyPoint> keypoints1;
    vector <KeyPoint> keypoints2;


    // Construction of the SURF feature detector
    SurfFeatureDetector surf(2500.);

    // Detection of the SURF features
    surf.detect(img1,keypoints1);
    surf.detect(img2,keypoints2);

    cout << "Number of SURF points 1 : " << keypoints1.size() << endl;
    cout << "Number of SURF points 2 : " << keypoints2.size() << endl;

    // Draw a key points
    drawKeypoints(img1,keypoints1,img1o,Scalar(255,0,0),DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    imshow("testc1_features",img1o);
    drawKeypoints(img2,keypoints2,img2o,Scalar(255,0,0),DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    imshow("testc2_features",img2o);

    // Construction of the SURF descriptor extractor
    SurfDescriptorExtractor surfDesc;
    // Extraction of the SURF descriptors
    Mat descriptors1,descriptors2;

    surfDesc.compute(img1,keypoints1,descriptors1);
    surfDesc.compute(img2,keypoints2,descriptors2);

    cout << "descriptor1 matrix size: " << descriptors1.rows << " by " << descriptors1.cols << endl;
    cout << "descriptor2 matrix size: " << descriptors2.rows << " by " << descriptors2.cols << endl;


    // Construction of the matcher
    BruteForceMatcher< L2 <float> > matcher;


    // Match the two image descriptors
    vector<DMatch> matches;
    matcher.match(descriptors1,descriptors2, matches);

    cout << "Number of matched points: " << matches.size() << endl;

    // Select few Matches

    vector<DMatch> selMatches;

    selMatches.push_back(matches[7]);
    selMatches.push_back(matches[6]);
    selMatches.push_back(matches[10]);
    selMatches.push_back(matches[5]);
    selMatches.push_back(matches[43]);
    selMatches.push_back(matches[25]);
    selMatches.push_back(matches[34]);
    // Draw the selected matches
    Mat imageMatches;
    drawMatches(img1,keypoints1,img2,keypoints2,selMatches,imageMatches,Scalar(255,255,150));
    imshow("Matches",imageMatches);

    // Convert 1 vector of keypoints into 2 vector of point2f

    vector<int> pointIndexes1;
    vector<int> pointIndexes2;

    for (vector<DMatch>::const_iterator it= selMatches.begin();
                   it!= selMatches.end(); ++it) {

                           // Get the indexes of the selected matched keypoints
                           pointIndexes1.push_back(it->queryIdx);
                           pointIndexes2.push_back(it->trainIdx);
          }

    //Convert keypoints into Point2f

    vector<Point2f> selPoints1, selPoints2;

   KeyPoint::convert(keypoints1,selPoints1,pointIndexes1);
   KeyPoint::convert(keypoints2,selPoints2,pointIndexes2);

   vector<Point2f>::const_iterator it= selPoints1.begin();
          while (it!=selPoints1.end()) {

                  // draw a circle at each corner location
                  circle(img1,*it,3,Scalar(255,255,255),2);
                  ++it;
          }

          it= selPoints2.begin();
          while (it!=selPoints2.end()) {

                  // draw a circle at each corner location
                  circle(img2,*it,3,Scalar(255,255,255),2);
                  ++it;
          }

    // Compute F matrix from 7 matches
       cv::Mat fundemental= cv::findFundamentalMat(
                    cv::Mat(selPoints1), // points in first image
                    cv::Mat(selPoints2), // points in second image
                    CV_RANSAC);       // 7-point method

    std::cout << "F-Matrix size= " << fundemental.rows << "," << fundemental.cols << std::endl;

    // draw the left points corresponding epipolar lines in right image

    vector<Vec3f>lines1;
    computeCorrespondEpilines(Mat(selPoints1),1,fundemental,lines1);


    for (vector<Vec3f>::const_iterator it= lines1.begin();
                     it!=lines1.end(); ++it) {

                             // draw the epipolar line between first and last column
                             line(img2,Point(0,-(*it)[2]/(*it)[1]),
                                                 Point(img2.cols,-((*it)[2]+(*it)[0]*img2.cols)/(*it)[1]),
                                                             Scalar(255,255,255));
            }



    // draw the left points corresponding epipolar lines in left image
            vector<Vec3f> lines2;
            computeCorrespondEpilines(Mat(selPoints2),2,fundemental,lines2);
            for (vector<Vec3f>::const_iterator it= lines2.begin();
                     it!=lines2.end(); ++it) {

                             // draw the epipolar line between first and last column
                             line(img1,Point(0,-(*it)[2]/(*it)[1]),
                                                 Point(img1.cols,-((*it)[2]+(*it)[0]*img1.cols)/(*it)[1]),
                                                             Scalar(255,255,255));
            }

        // Display the images with points and epipolar lines
            cv::namedWindow("Right Image Epilines");
            cv::imshow("Right Image Epilines",img1);
            cv::namedWindow("Left Image Epilines");
            cv::imshow("Left Image Epilines",img2);

            std::vector<cv::Point2f> points1, points2;
                   for (std::vector<cv::DMatch>::const_iterator it= matches.begin();
                            it!= matches.end(); ++it) {

                                    // Get the position of left keypoints
                                    float x= keypoints1[it->queryIdx].pt.x;
                                    float y= keypoints1[it->queryIdx].pt.y;
                                    points1.push_back(cv::Point2f(x,y));
                                    // Get the position of right keypoints
                                    x= keypoints2[it->trainIdx].pt.x;
                                    y= keypoints2[it->trainIdx].pt.y;
                                    points2.push_back(cv::Point2f(x,y));
                   }

                   std::cout << points1.size() << " " << points2.size() << std::endl;

                   // Compute F matrix using RANSAC
                           std::vector<uchar> inliers(points1.size(),0);
                           fundemental= cv::findFundamentalMat(
                                   cv::Mat(points1),cv::Mat(points2), // matching points
                                   inliers,      // match status (inlier ou outlier)
                                   CV_FM_RANSAC, // RANSAC method
                                   1,            // distance to epipolar line
                                   0.98);        // confidence probability



    Mat image1 = imread("testc1.png", CV_LOAD_IMAGE_COLOR);
    Mat image2= imread("testc2.png", CV_LOAD_IMAGE_COLOR);

    // Draw the epipolar line of few points
            cv::computeCorrespondEpilines(cv::Mat(selPoints1),1,fundemental,lines1);
            for (vector<cv::Vec3f>::const_iterator it= lines1.begin();
                     it!=lines1.end(); ++it) {

                             cv::line(image2,cv::Point(0,-(*it)[2]/(*it)[1]),
                                                 cv::Point(image2.cols,-((*it)[2]+(*it)[0]*image2.cols)/(*it)[1]),
                                                             cv::Scalar(255,255,255));
            }

            cv::computeCorrespondEpilines(cv::Mat(selPoints2),2,fundemental,lines2);
            for (vector<cv::Vec3f>::const_iterator it= lines2.begin();
                     it!=lines2.end(); ++it) {

                             cv::line(image1,cv::Point(0,-(*it)[2]/(*it)[1]),
                                                 cv::Point(image1.cols,-((*it)[2]+(*it)[0]*image1.cols)/(*it)[1]),
                                                             cv::Scalar(255,255,255));
            }

            // Draw the inlier points
            std::vector<cv::Point2f> points1In, points2In;
            std::vector<cv::Point2f>::const_iterator itPts= points1.begin();
            std::vector<uchar>::const_iterator itIn= inliers.begin();
            while (itPts!=points1.end()) {

                    // draw a circle at each inlier location
                    if (*itIn) {
                            cv::circle(image1,*itPts,3,cv::Scalar(255,255,255),2);
                            points1In.push_back(*itPts);
                    }
                    ++itPts;
                    ++itIn;
            }

            itPts= points2.begin();
            itIn= inliers.begin();
            while (itPts!=points2.end()) {

                    // draw a circle at each inlier location
                    if (*itIn) {
                            cv::circle(image2,*itPts,3,cv::Scalar(255,255,255),2);
                            points2In.push_back(*itPts);
                    }
                    ++itPts;
                    ++itIn;
            }

        // Display the images with points
            cv::namedWindow("Right Image Epilines (RANSAC)");
            cv::imshow("Right Image Epilines (RANSAC)",image1);
            cv::namedWindow("Left Image Epilines (RANSAC)");
            cv::imshow("Left Image Epilines (RANSAC)",image2);

            cv::findHomography(cv::Mat(points1In),cv::Mat(points2In),inliers,CV_RANSAC,1.);

            // Read input images
            image1= imread("testc1.png", CV_LOAD_IMAGE_COLOR);
            image2= imread("testc2.png", CV_LOAD_IMAGE_COLOR);

            // Draw the inlier points
            itPts= points1In.begin();
            itIn= inliers.begin();
            while (itPts!=points1In.end()) {

                    // draw a circle at each inlier location
                    if (*itIn)
                            cv::circle(image1,*itPts,3,cv::Scalar(255,255,255),2);

                    ++itPts;
                    ++itIn;
            }

            itPts= points2In.begin();
            itIn= inliers.begin();
            while (itPts!=points2In.end()) {

                    // draw a circle at each inlier location
                    if (*itIn)
                            cv::circle(image2,*itPts,3,cv::Scalar(255,255,255),2);

                    ++itPts;
                    ++itIn;
            }

        // Display the images with points
            cv::namedWindow("Right Image Homography (RANSAC)");
            cv::imshow("Right Image Homography (RANSAC)",image1);
            cv::namedWindow("Left Image Homography (RANSAC)");
            cv::imshow("Left Image Homography (RANSAC)",image2);

            cv::waitKey();




    return a.exec();

}
