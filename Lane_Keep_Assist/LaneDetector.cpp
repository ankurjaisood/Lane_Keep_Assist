#include<opencv2/opencv.hpp>
#include<iostream>
#include<filesystem>

// Includes
namespace fs = std::filesystem;


// Lane Detector Class

class LaneDetector {
private:
	cv::Mat calibration_matrix;
	cv::Mat distance_coefficients;
	bool calibrated;

	void calibrate_camera(std::string camera_calibration_images, cv::Size chessboard_size, bool debug = false) {

		// Initialize arrays to hold points
		std::vector<std::vector<cv::Point3f>> object_points;
		std::vector<std::vector<cv::Point2f>> image_points;

		// Populate object points
		int numSquares = chessboard_size.width * chessboard_size.height;
		std::vector<cv::Point3f> obj;
		for (int j = 0; j < numSquares; j++)
			obj.push_back(cv::Point3f(j / chessboard_size.width, j % chessboard_size.width, 0.0f));

		// Initialize variable to hold image size
		cv::Size image_size = cv::Size(0, 0);

		// Read all images from cal folder
		for (const auto& image_path : fs::directory_iterator(camera_calibration_images)) {

			// Get the string path
			std::string image_string = image_path.path().string();

			// Read the image matrix
			cv::Mat image = cv::imread(image_string);
			cv::Mat gray_image;
			cv::cvtColor(image, gray_image, cv::COLOR_BGR2GRAY);

			// Get the size of the image
			if (image_size == cv::Size(0, 0)) {
				image_size = image.size();
			}

			// Find the chessboard corners
			std::vector<cv::Point2f> corners;
			bool corners_found = cv::findChessboardCorners(gray_image, chessboard_size, corners);

			if (corners_found) {
				image_points.push_back(corners);
				object_points.push_back(obj);

				if (debug) {
					cv::drawChessboardCorners(image, chessboard_size, corners, corners_found);
					// Show the image
					cv::imshow(image_string, image);
					cv::waitKey(500);
				}

			}
		}

		// Calibrate camera
		cv::Mat cal_matrix;
		cv::Mat dist_coeffs;
		std::vector<cv::Mat> r_vecs;
		std::vector<cv::Mat> t_vecs;
		cv::calibrateCamera(object_points, image_points, image_size, cal_matrix, dist_coeffs, r_vecs, t_vecs);

		// Set calibration matrix
		calibration_matrix = cal_matrix;
		distance_coefficients = dist_coeffs;
		calibrated = true;
	}

	cv::Mat undistort_image(cv::Mat image) {
		cv::Mat undistorted_image;
		if (calibrated) {
			cv::undistort(image, undistorted_image, calibration_matrix, distance_coefficients);
		}
		return undistorted_image;
	}

public:
	LaneDetector() {
		calibrated = false;
	}

	LaneDetector(std::string camera_calibration_images, cv::Size chessboard_size, bool debug=false) {
		calibrated = false;

		// Calibrate camera
		calibrate_camera(camera_calibration_images, chessboard_size, debug);
	}
};