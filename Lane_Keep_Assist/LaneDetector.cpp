#include<opencv2/opencv.hpp>
#include<iostream>
#include<filesystem>
#include"Lane.cpp"

// Namespaces
namespace fs = std::filesystem;

// Lane Detector Class

class LaneDetector {
private:
	Lane left_lane;
	Lane right_lane;

	cv::Mat calibration_matrix;
	cv::Mat distance_coefficients;
	cv::Mat transform_matrix;
	cv::Mat inv_transform_matrix;

	const cv::Point2f src_points[4] = { cv::Point2f(150, 700), cv::Point2f(450, 500),
		cv::Point2f(850, 500), cv::Point2f(1200, 700) };

	bool transform_calculated;
	bool calibrated;

	const int GRADIENT_THRES_MIN = 30;
	const int GRADIENT_THRES_MAX = 100;
	const int COLOUR_THRES_MIN = 180;
	const int COLOUR_THRES_MAX = 255;
	const int OFFSET = 100;
	
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
				//std::cout << "Calibrating with image: " + image_string << std::endl;

				if (debug) {
					cv::drawChessboardCorners(image, chessboard_size, corners, corners_found);
					// Show the image
					cv::imshow(image_string, image);
					cv::waitKey(500);
				}

			}
			else {
				// Tracking output
				//std::cout << "Skipped calibrating with image: " + image_string << std::endl;
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

	cv::Mat* undistort_image(cv::Mat &image, std::string &image_path) {
		cv::Mat* undistorted_image = new cv::Mat();;
		if (calibrated) {
			cv::undistort(image, *undistorted_image, calibration_matrix, distance_coefficients);
			// Tracking output
			//std::cout << "Undistoted image: " + image_path << std::endl;
		}
		return undistorted_image;
	}

	cv::Mat* threshold_image(cv::Mat &image, std::string &image_path) {
		// Create an output image
		cv::Mat* thresholded_image = new cv::Mat();;

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
		cv::bitwise_or(thresholded_sobelx, thresholded_schannel, *thresholded_image);

		// Tracking output
		//std::cout << "Scaled and Thresholded image: " + image_path << std::endl;

		return thresholded_image;
	}

	void get_perspective_transform(cv::Mat& image, std::string& image_path) {
		// Set source and destination points
		cv::Size img_size = image.size();

		// get dst points
		cv::Point2f dst_points[4] = { cv::Point2f(OFFSET, img_size.height - OFFSET), cv::Point2f(OFFSET, OFFSET),
			cv::Point2f(img_size.width - OFFSET, OFFSET), cv::Point2f(img_size.width - OFFSET, img_size.height - OFFSET) };


		// Calculate perspective transform matrix
		transform_matrix = cv::getPerspectiveTransform(src_points, dst_points);
		inv_transform_matrix = cv::getPerspectiveTransform(dst_points, src_points);
		transform_calculated = true;

		// Tracking output
		//std::cout << "Calculated perspective transform using: " + image_path << std::endl;
	}

	cv::Mat* perspective_transform(cv::Mat &image, std::string &image_path) {
		// Transform the image using the calculated matrix
		cv::Mat* transformed_image = new cv::Mat();;
		cv::warpPerspective(image, *transformed_image, transform_matrix, image.size());

		// Tracking output
		//std::cout << "Transformed image: " + image_path << std::endl;

		return transformed_image;
	}

	cv::Mat* inv_perspective_transform(cv::Mat& image, std::string& image_path) {
		// Transform the image using the calculated matrix
		cv::Mat* inv_transformed_image = new cv::Mat();
		cv::warpPerspective(image, *inv_transformed_image, inv_transform_matrix, image.size());

		// Tracking output
		//std::cout << "Inverse Transformed image: " + image_path << std::endl;

		return inv_transformed_image;
	}

	cv::Mat* detect_lanes(cv::Mat &original_image, cv::Mat &image, std::string &image_path) {
		// Prepare output
		cv::Mat* detected_lanes = new cv::Mat();;

		// Fine all nonzero pixels
		std::vector<cv::Point> nonzero;
		cv::findNonZero(image, nonzero);

		// Get the bottom half of the image
		cv::Mat bot_half = image(cv::Rect(0, image.rows / 2, image.cols, image.rows / 2));

		// Get left and right halves of bottom half of image
		cv::Mat left_half = bot_half(cv::Rect(0, 0, bot_half.cols / 2, bot_half.rows));
		cv::Mat right_half = bot_half(cv::Rect(bot_half.cols / 2, 0, bot_half.cols / 2, bot_half.rows));

		// Get "histogram" of each side
		cv::Mat left_hist;
		cv::Mat right_hist;
		cv::reduce(left_half, left_hist, 0, cv::REDUCE_SUM, CV_32F);
		cv::reduce(right_half, right_hist, 0, cv::REDUCE_SUM, CV_32F);
		
		// Extract max values from histogram halves
		cv::Point left_base;
		cv::Point right_base;

		cv::minMaxLoc(left_hist, 0, 0, 0, &left_base);
		cv::minMaxLoc(right_hist, 0, 0, 0, &right_base);

		// Sliding window for each side of image
		cv::Point left_current = left_base;
		cv::Point right_current = right_base;

		// Sliding window and polynomial fit
		std::vector<double> left_fit = left_lane.find_lane(image, nonzero, left_current);
		std::vector<double> right_fit = right_lane.find_lane(image, nonzero, right_current);

		// Create points for fillpoly
		std::vector<cv::Point> l_points;
		std::vector<cv::Point> r_points;

		for (int i = 0; i < image.size().height; ++i) {
			int left_x = int(left_fit[0] + left_fit[1] * i + pow(left_fit[2] * i, 2));
			int right_x = int(right_fit[0] + right_fit[1] * i + pow(right_fit[2] * i, 2));
			l_points.push_back(cv::Point(left_x, i));
			r_points.push_back(cv::Point(right_x, i));
		}
		l_points.insert(l_points.end(), r_points.begin(), r_points.end());
		std::vector<std::vector<cv::Point>> points = { l_points };

		// Draw lanes on image
		cv::fillPoly(original_image, points, cv::Scalar(255, 255, 255));

		//std::cout << "Detected Lanes: " + image_path << std::endl;

		return &original_image;
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

	cv::Mat* find_lanes(cv::Mat &image, std::string &image_path) {

		// Output image
		cv::Mat* resultant_image;

		// Undistort image
		cv::Mat* undistorted_image = undistort_image(image, image_path);

		if (!transform_calculated) {
			// Calculate perspective transform
			get_perspective_transform(image, image_path);
		}

		// Colour and gradient threshold
		cv::Mat* thresholded_image = threshold_image(*undistorted_image, image_path);

		// Get perspective transform
		cv::Mat* transformed_image = perspective_transform(*thresholded_image, image_path);

		// Find lanes
		cv::Mat* lane_image = detect_lanes(image, *transformed_image, image_path);

		// Inverse perspective transform
		resultant_image = inv_perspective_transform(*lane_image, image_path);

		return resultant_image;
	}
};