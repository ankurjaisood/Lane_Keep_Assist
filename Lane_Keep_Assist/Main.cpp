#include<opencv2/opencv.hpp>
#include<iostream>
#include<filesystem>
#include "LaneDetector.cpp"

// ENVIORNMENT VARIABLES
std::string TEST_IMAGES_DIRECTORY = "resources/test_images/";
std::string TEST_IMAGES_OUTPUT_DIRECTORY = "resources/test_images_output/";
std::string CAMERA_CALIBRATION_IMAGES = "resources/camera_cal/";
cv::Size CHESSBOARD_SIZE = cv::Size(9, 6);

// NAMESPACE VARIABLES
namespace fs = std::filesystem;

// TEST FUNCTIONS
void test_sample_images() {
    // Create a LaneDetector object
    LaneDetector detector = LaneDetector(CAMERA_CALIBRATION_IMAGES, CHESSBOARD_SIZE);

    // Get all images in test images directory
    for (const auto& image_path : fs::directory_iterator(TEST_IMAGES_DIRECTORY)) {

        // Get string version of path
        std::string image_string = image_path.path().string();

        // Read the image matrix
        cv::Mat image = cv::imread(image_string);

        // Undistort the image
        cv::Mat undistorted_image = detector.undistort_image(image);

        // Get filename
        fs::path pathObj(image_path);
        std::string file_name = pathObj.filename().string();
        std::string output_path = TEST_IMAGES_OUTPUT_DIRECTORY + file_name;
        
        // Write the resultant image
        cv::imwrite(output_path, undistorted_image);

        // Display image
        //cv::imshow(image_string, image);
        //cv::waitKey(500);
    }

}

// START OF MAIN PROGRAM 
int main()
{
    test_sample_images();
    return 0;
}
