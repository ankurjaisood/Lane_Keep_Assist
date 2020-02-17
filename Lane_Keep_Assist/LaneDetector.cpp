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
	cv::Mat transform_matrix;
	cv::Mat inv_transform_matrix;

	bool transform_calculated;
	bool calibrated;

	int GRADIENT_THRES_MIN = 30;
	int GRADIENT_THRES_MAX = 100;
	int COLOUR_THRES_MIN = 180;
	int COLOUR_THRES_MAX = 255;
	int OFFSET = 100;

	void calibrate_camera(std::string &camera_calibration_images, cv::Size &chessboard_size, bool debug = false) {

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
			cv::Mat gray_image = image;
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

				// Tracking output
				std::cout << "Calibrating with image: " + image_string << std::endl;

				if (debug) {
					cv::drawChessboardCorners(image, chessboard_size, corners, corners_found);
					// Show the image
					cv::imshow(image_string, image);
					cv::waitKey(500);
				}

			}
			else {
				// Tracking output
				std::cout << "Skipped calibrating with image: " + image_string << std::endl;
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

	cv::Mat undistort_image(cv::Mat &image, std::string &image_path) {
		cv::Mat undistorted_image;
		if (calibrated) {
			cv::undistort(image, undistorted_image, calibration_matrix, distance_coefficients);
			// Tracking output
			std::cout << "Undistoted image" + image_path << std::endl;
		}
		return undistorted_image;
	}

	cv::Mat threshold_image(cv::Mat &image, std::string &image_path) {
		// Create an output image
		cv::Mat thresholded_image;

		// Convert to HLS colour space
		cv::Mat hls_image;
		cv::cvtColor(image, hls_image, cv::COLOR_RGB2HLS);

		// Convert to grayscale
		cv::Mat gray_image;
		cv::cvtColor(image, gray_image, cv::COLOR_BGR2GRAY);

		// Sobel x, y
		int kernel_size = 9;
		cv::Mat sobelx;
		cv::Sobel(gray_image, sobelx, CV_64F, 1, 0, kernel_size);

		// Scale sobel
		cv::Mat scaled_sobelx;
		double minVal, maxVal;
		cv::minMaxLoc(sobelx, &minVal, &maxVal);
		sobelx.convertTo(scaled_sobelx, CV_8U, 255.0 / (maxVal - minVal), -minVal * 255.0 / (maxVal - minVal));

		// Apply threshold on sobelx
		cv::Mat thresholded_sobelx;
		cv::inRange(scaled_sobelx, GRADIENT_THRES_MIN, GRADIENT_THRES_MAX, thresholded_sobelx);

		// Extract S channel from HLS image
		cv::Mat channels[3];
		cv::split(hls_image, channels);
		cv::Mat s_channel = channels[2];

		// Apply threshold on s channel
		cv::Mat thresholded_schannel;
		cv::inRange(s_channel, COLOUR_THRES_MIN, COLOUR_THRES_MAX, thresholded_schannel);

		// Combine both channels
		cv::bitwise_or(thresholded_sobelx, thresholded_schannel, thresholded_image);

		// Tracking output
		std::cout << "Scaled and Thresholded image" + image_path << std::endl;

		return thresholded_image;
	}

	void get_perspective_transform(cv::Mat& image, std::string& image_path) {
		// Set source and destination points
		cv::Size img_size = image.size();


		// Calculate perspective transform matrix
		transform_matrix = cv::getPerspectiveTransform();
		inv_transform_matrix = cv::getPerspectiveTransform();

		/*
		src = np.float32(SRC_POINTS)
		dst = np.float32([[OFFSET, img_size[1] - OFFSET], [OFFSET, OFFSET], [img_size[0] - OFFSET, OFFSET], [img_size[0] - OFFSET, img_size[1] - OFFSET]] )

		// Calculate perspective transform matrix
		transform_matrix = cv::getPerspectiveTransform(src, dst);
		inv_transform_matrix = cv2.getPerspectiveTransform(dst, src)
		*/


	}

	cv::Mat perspective_transform(cv::Mat &image, std::string &image_path) {
		cv::Mat transformed_image = image;

		return transformed_image;
	}

	cv::Mat detect_lanes(cv::Mat &image, std::string &image_path) {
		cv::Mat detected_lanes = image;

		return detected_lanes;
	}

public:
	LaneDetector() {
		calibrated = false;
		transform_calculated = false;
	}

	LaneDetector(std::string &camera_calibration_images, cv::Size &chessboard_size, bool debug=false) {
		calibrated = false;
		transform_calculated = false;

		// Calibrate camera
		calibrate_camera(camera_calibration_images, chessboard_size, debug);
	}

	cv::Mat find_lanes(cv::Mat &image, std::string &image_path) {

		// Copy image
		cv::Mat resultant_image;

		// Undistort image
		cv::Mat undistorted_image = undistort_image(image, image_path);

		if (!transform_calculated) {
			// Calculate perspective transform
			get_perspective_transform(image, image_path);
		}

		// Colour and gradient threshold
		cv::Mat thresholded_image = threshold_image(undistorted_image, image_path);

		// Get perspective transform
		cv::Mat transformed_image = perspective_transform(thresholded_image, image_path);

		// Find lanes
		resultant_image = detect_lanes(transformed_image, image_path);

		return resultant_image;
	}
};