#include "opencv2/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/aruco.hpp"
#include "opencv2/calib3d.hpp"
#include <opencv2/calib3d/calib3d_c.h>
#include <opencv2/highgui/highgui_c.h>

#include <sstream>
#include <iostream>
#include <fstream>

using namespace std;
using namespace cv;

const float calibrationSquareDimension = 0.02474f; //the sidelength of the individual squares on the printed checkard board in meters
const float arucoMarkerSquareDimension = 0.01080f; //side length of the printed Aruco Markers 10.8 mm
const Size checkardboardDimensions = Size(6, 9);

//A function that generates 12 unique Aruco markers using the 6x6_50 predefined ArUco marker dictionary
void generateAruco()
{
	Mat outputAruco;
	Ptr<aruco::Dictionary> arucoDictionary = aruco::getPredefinedDictionary(aruco::PREDEFINED_DICTIONARY_NAME::DICT_6X6_50);

	for (int i = 0; i < 12; i++)
	{
		aruco::drawMarker(arucoDictionary, i, 42, outputAruco, 1);
		ostringstream convert;
		string imagename = "6x6ArucoMarker_";

		imwrite(convert.str(), outputAruco);
	}
}

//A function that creates known calibration checkerboard position
void createKnownBoardPosition(Size boardSize, float squareEdgeLength, vector<Point3f>& corners)
{
	for (int i = 0; i < boardSize.height; i++)
	{
		for (int j = 0; j < boardSize.width; j++)
		{
			corners.push_back(Point3f(j * squareEdgeLength, i * squareEdgeLength, 0.0f));
		}
	}

}

//A function that gets the corner locations of the calibration chekardboard
void getChekardBoardCorners(vector<Mat> images, vector<vector<Point2f>>& allFoundCorners, bool showResults = false)
{
	for (vector<Mat>::iterator iter = images.begin(); iter != images.end(); iter++)
	{
		vector<Point2f> pointBuf;
		bool found = findChessboardCorners(*iter, checkardboardDimensions, pointBuf, CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_NORMALIZE_IMAGE);

		if (found)
		{
			allFoundCorners.push_back(pointBuf);
		}

		if (showResults)
		{
			drawChessboardCorners(*iter, checkardboardDimensions, pointBuf, found);
			imshow("Looking for Corners", *iter);
			waitKey(0);
		}
	}
}

//A function which continually monitors the live webcam stream for any of the generated ArUco Markers, and determines the translation and location offset in world space where they are located
int startWebcamMonitoring(const Mat& cameraMatrix, const Mat& distanceCoefficients, float arucoMarkerSquareDimension)
{
	Mat frame; //holds the frame of information from the webcam, the frame we will analyze to see if an ArUco marker is present

	vector<int> markerIds; // vector of integers that holds are marker Ids
	vector<vector<Point2f>> markerCorners, rejectedCandidates;
	aruco::DetectorParameters parameters; //built in arucolibrary function to find parameters for detection

	Ptr <aruco::Dictionary> markerDictionary = aruco::getPredefinedDictionary(aruco::PREDEFINED_DICTIONARY_NAME::DICT_6X6_50);

	//check if the video webcam capture source is open, give it a name, open up a window and continually loop over a window looking for the ArUco markers

	VideoCapture vid(0);

	if (!vid.isOpened()) // if the video capture is not opened exit
	{
		return -1;
	}

	namedWindow("Webcam", CV_WINDOW_AUTOSIZE);
	vector <Vec3d> rotationVectors, translationVectors;

	while (true)
	{
		if (!vid.read(frame))// if we cannot read a frame from the loop exit
			break;

		aruco::detectMarkers(frame, markerDictionary, markerCorners, markerIds); 
		aruco::drawDetectedMarkers(frame, markerCorners, markerIds, Scalar(0,255,0));
		aruco::estimatePoseSingleMarkers(markerCorners, arucoMarkerSquareDimension, cameraMatrix, distanceCoefficients, rotationVectors, translationVectors);
		
		for (int i = 0; i < markerIds.size(); i++)
		{
			aruco::drawAxis(frame, cameraMatrix, distanceCoefficients, rotationVectors[i], translationVectors[i], 0.1f);
		}
		imshow("Webcam", frame);
		if (waitKey(30) >= 0)
		{
			break;
		}	
	}

	return 1;
}

//camera calibration function
void cameraCalibration(vector<Mat> calibrationImages, Size boardSize, float squareEdgeLength, Mat& cameraMatrix, Mat& distanceCore)
{
	vector<vector<Point2f>> checkardboardImageSpacePoints;
	getChekardBoardCorners(calibrationImages, checkardboardImageSpacePoints, false);

	vector<vector<Point3f>> worldSpaceCornerPoints(1);

	createKnownBoardPosition(boardSize, squareEdgeLength, worldSpaceCornerPoints[0]);
	worldSpaceCornerPoints.resize(checkardboardImageSpacePoints.size(), worldSpaceCornerPoints[0]); // relationship between the 2D points and the 3D points that we expect, continuous copying

	vector<Mat> rVectors, tVectors; // radial and tangential vectors

	Mat distanceCoefficients; // added

	distanceCoefficients = Mat::zeros(8, 1, CV_64F);

	calibrateCamera(worldSpaceCornerPoints, checkardboardImageSpacePoints, boardSize, cameraMatrix, distanceCoefficients, rVectors, tVectors);

}

bool saveCameraCalibration(string name, Mat cameraMatrix, Mat distanceCoefficients)
{
	ofstream outStream(name);

	if (outStream) // if we have a stream do the following, push out rows and columns
	{
		uint16_t rows = cameraMatrix.rows;
		uint16_t columns = cameraMatrix.cols;

		outStream << rows << endl;
		outStream << columns << endl;

		for (int r = 0; r < rows; r++)
		{
			for (int c = 0; c < columns; c++)
			{
				double value = cameraMatrix.at<double>(r, c); // take the temporary value out of the cameraMatrix and put in temp variable
				outStream << value << endl; //push it out to the stream
			}
		}
		rows = distanceCoefficients.rows;
		columns = distanceCoefficients.cols;

		outStream << rows << endl;
		outStream << columns << endl; //push out rows and columns

		for (int r = 0; r < rows; r++)
		{
			for (int c = 0; c < columns; c++)
			{
				double value = distanceCoefficients.at<double>(r, c);
				outStream << value << endl;
			}
		}

		outStream.close();
		return true;
	}

	return false;
}

//A function to load the cameraMatrix information we put into the .txt file when we took pictures to calibrate
bool loadCameraCalibration(string name, Mat& cameraMatrix, Mat& distanceCoefficients)
{
	ifstream inStream(name);
	if (inStream)
	{
		uint16_t rows;
		uint16_t columns;

		inStream >> rows;
		inStream >> columns;

		cameraMatrix = Mat(Size(columns, rows), CV_64F);

		for (int r = 0; r < rows; r++)
		{
			for (int c = 0; c < columns; c++)
			{
				double temp = 0.0f;
				inStream >> temp; //push info onto temporary variable
				cameraMatrix.at<double>(r, c) = temp;
				cout << cameraMatrix.at<double>(r, c) << "\n";
			}
		}

		//Grabbing Distance Coefficients
		inStream >> rows;
		inStream >> columns;

		distanceCoefficients = Mat::zeros(rows, columns, CV_64F);

		for (int r = 0; r < rows; r++)
		{
			for (int c = 0; c < columns; c++)
			{
				double temp = 0.0f;
				inStream >> temp;
				distanceCoefficients.at<double>(r, c) = temp;
				cout << distanceCoefficients.at<double>(r, c) << "\n";
			}
		}
		inStream.close();
		return true;
	}

	return false;
}

void cameraCalibrationProcess(Mat& cameraMatrix, Mat& distanceCoefficients)
{
	Mat frame;
	Mat drawToFrame;

	vector<Mat> savedImages;
	vector<vector<Point2f>> markerCorners, rejectedCandidates;

	VideoCapture vid(0);

	if (!vid.isOpened())
	{
		return;
	}

	int framesPerSecond = 24; //fps of recording check webcam specs

	namedWindow("Webcam", CV_WINDOW_AUTOSIZE);

	while (true) 
	{
		if (!vid.read(frame))
		{
			break;
		}

		vector<Vec2f> foundPoints;
		bool found = false;

		found = findChessboardCorners(frame, checkardboardDimensions, foundPoints, CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_NORMALIZE_IMAGE); // |CV_CALIB_CB_FAST_CHECK
		frame.copyTo(drawToFrame);
		drawChessboardCorners(drawToFrame, checkardboardDimensions, foundPoints, found); // draw the cheackrdboard corners it detects

		if (found)
		{
			imshow("Webcam", drawToFrame);
		}

		else
			imshow("Webcam", frame);
		char character = waitKey(1000 / framesPerSecond);

		switch (character)
		{
		case ' ':
			// saving the image whenever the user hits the space bar
			if (found)
			{
				Mat temp;
				frame.copyTo(temp);
				savedImages.push_back(temp);
			}
			break;
		case 13:
			//start calibration whenever the user hits the enter key
			if (savedImages.size() > 15) // make sure that we have enough valid images (greater than 10)
			{
				cameraCalibration(savedImages, checkardboardDimensions, calibrationSquareDimension, cameraMatrix, distanceCoefficients);
				saveCameraCalibration("TheCameraCalibrationFile", cameraMatrix, distanceCoefficients);
			}
			break;
		case 27:
			//exit the program when the user hits the esc key
			return;
			break;
		}
	}
}

int main(int argv, char** argc)
{
	Mat cameraMatrix = Mat::eye(3, 3, CV_64F);

	Mat distanceCoefficients;

	//uncomment the cameraCalibration process function below if you need to calibrate the webccam with the checkerboard, comment the remaining functions out, once the calibration process is complete uncomment the remaining functions and comment the calibration function 
	//cameraCalibrationProcess(cameraMatrix, distanceCoefficients); 

	loadCameraCalibration("TheCameraCalibrationFile", cameraMatrix, distanceCoefficients);

	startWebcamMonitoring(cameraMatrix, distanceCoefficients, arucoMarkerSquareDimension);

	return 0;
}