#include<opencv2/opencv.hpp>
#include<vector>
#include<stdio.h>
#include<stdlib.h>
#include<assert.h>
#include<cuda_runtime.h>
#include<device_launch_parameters.h>

__device__ int win_y_low = 100;
__device__ int win_y_high = 200;
__device__ int win_x_low = 100;
__device__ int win_x_high = 200;
#define numofpoints 50000000

// CUDA kernel to check points
struct my_point {
    int x;
    int y;
    my_point(int myx, int myy) : x(myx), y(myy) {}
    my_point() {}
};

__device__ my_point dev_pts[numofpoints];
__device__ int dev_count = 0;

__device__ int atomic_push_back(my_point& p) {
    int insert_pt = atomicAdd(&dev_count, 1);
    if (insert_pt < numofpoints) {
        dev_pts[insert_pt].x = p.x;
        dev_pts[insert_pt].y = p.y;
        return insert_pt;
    }
    else return -1;
}

__global__ void check_valid_point(const my_point* pts, int numPoints) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;

	if (i < numPoints)
	{
        // Check to see if valid point
        my_point p;
        p.x = pts[i].x;
        p.y = pts[i].y;

        if ((win_x_low < p.x) && (p.x < win_x_high) && (win_y_low < p.y) && (p.y < win_y_high)) {
            // If valid point atomic add to output array
            atomic_push_back(p);
        }
	}
}

int finding_lane_pixels_cpp(std::vector<cv::Point> pts) {
	std::vector<cv::Point> line_points;

	for (auto p : pts) {
		if ((win_x_low < p.x) && (p.x < win_x_high) && (win_y_low < p.y) && (p.y < win_y_high)) {
			line_points.push_back(p);
		}
	}

	return line_points.size();
}

int finding_lane_pixels_cuda(my_point pts[]) {

    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;

    // Compute vector size
    size_t size = numofpoints * sizeof(my_point);

    // Allocate the host input vector A
    my_point* h_pts = &pts[0];

    // Verify that allocations succeeded
    if (h_pts == NULL)
    {
        fprintf(stderr, "Failed to allocate host vector!\n");
        exit(EXIT_FAILURE);
    }

    // Allocate the device input vector
    my_point* d_pts = NULL;
    err = cudaMalloc((void**)&d_pts, size);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Copy the host input vector in host memory to the device input vector in device memory
    err = cudaMemcpy(d_pts, h_pts, size, cudaMemcpyHostToDevice);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Launch the Vector Add CUDA Kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (numofpoints + threadsPerBlock - 1) / threadsPerBlock;
    check_valid_point<<<blocksPerGrid, threadsPerBlock>>> (d_pts, numofpoints);
    err = cudaGetLastError();

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Copy the device result vector in device memory to the host result vector
    // in host memory.
    std::vector<my_point> line_points(numofpoints);
    err = cudaMemcpyFromSymbol(&(line_points[0]), dev_pts, size);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector C from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Free device global memory
    err = cudaFree(d_pts);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector A (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    };

	return sizeof(line_points);
}
extern "C" void run_tests() {
    // Create points
    std::vector<cv::Point>* points = new std::vector<cv::Point>(numofpoints);
    my_point* cuda_points = new my_point[numofpoints];
    for (int i = 0; i < numofpoints; ++i) {
        int x = rand() / (float)RAND_MAX;
        int y = rand() / (float)RAND_MAX;
        (*points)[i] = cv::Point(x, y);
        cuda_points[i] = my_point(x, y);
    }

    // Test 1
    auto start = cv::getTickCount() / cv::getTickFrequency();
    int cpp_found = finding_lane_pixels_cpp(*points);
    auto stop = cv::getTickCount() / cv::getTickFrequency();
    auto duration = (stop - start);
    std::cout << "Finding lane pixels using cpp: " << duration << std::endl;

    // Test 2
    start = cv::getTickCount() / cv::getTickFrequency();
    int cuda_found = finding_lane_pixels_cuda(cuda_points);
    stop = cv::getTickCount() / cv::getTickFrequency();
    duration = (stop - start);
    std::cout << "Finding lane pixels using cuda: " << duration << std::endl;

    assert(cpp_found == cuda_found);
}
