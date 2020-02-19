#include<opencv2/opencv.hpp>
#include<vector>
#include<stdio.h>
#include<stdlib.h>
#include<cuda_runtime.h>
#include<device_launch_parameters.h>

// CUDA kernel to check points
__global__ void check_valid_point(const float* A, const float* B, float* C, int numElements) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;

	if (i < numElements)
	{
		C[i] = A[i] + B[i];
	}
}

class CudaTests {
public:
	int win_y_low = 100;
	int win_y_high = 200;
	int win_x_low = 100;
	int win_x_high = 200;
	std::vector<cv::Point> points;

	CudaTests(int numofpoints) {
		for (int i = 0; i < numofpoints; ++i) {
			int x = rand() / (float)RAND_MAX;
			int y = rand() / (float)RAND_MAX;
			points[i] = cv::Point(x, y);
		}
	}

	void run_tests() {

		// Test 1
		auto start = cv::getTickCount() / cv::getTickFrequency();
		finding_lane_pixels_cpp(points);
		auto stop = cv::getTickCount() / cv::getTickFrequency();
		auto duration = (stop - start);
		std::cout << "Finding lane pixels using cpp: " << duration << std::endl;

		// Test 2
		start = cv::getTickCount() / cv::getTickFrequency();
		finding_lane_pixels_cpp(points);
		stop = cv::getTickCount() / cv::getTickFrequency();
		duration = (stop - start);
		std::cout << "Finding lane pixels using cuda: " << duration << std::endl;

	}

private:
	void finding_lane_pixels_cpp(std::vector<cv::Point> points) {
		int x_sum = 0;
		std::vector<cv::Point> line_points;

		for (auto p : points) {
			if ((win_x_low < p.x) && (p.x < win_x_high) && (win_y_low < p.y) && (p.y < win_y_high)) {
				line_points.push_back(p);
				x_sum += p.x;
			}
		}
	}
};