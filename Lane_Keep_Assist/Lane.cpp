#include<opencv2/opencv.hpp>
#include<vector>

// Class to represent a lane on the road

class Lane {
private:
	const int NUM_WINDOWS = 10;
	const int WINDOW_MARGIN = 100;
	const int MIN_NUMBER_PIXELS = 200;

	std::vector<cv::Point> sliding_window(cv::Mat& image, std::vector<cv::Point> nonzero, cv::Point current_base) {
		// Create a vector to store the poits which we will fit a polynomial to
		std::vector<cv::Point> poly_points;
		int window_height = image.size().height / NUM_WINDOWS;
		int x_current = current_base.x;

		for (int window_num = 0; window_num < NUM_WINDOWS; ++window_num) {
			// Create a vector to store the pixels within the region
			std::vector<cv::Point> line_points;

			// Create the window coordinates
			int win_y_low = image.size().height - (window_num + 1) * window_height;
			int win_y_high = image.size().height - window_num * window_height;
			int win_x_low = x_current - WINDOW_MARGIN;
			int win_x_high = x_current + WINDOW_MARGIN;

			// Draw the rectangle on the output image
			//cv::rectangle(image, cv::Point(win_x_low, win_y_low), cv::Point(win_x_high, win_y_high), cv::Scalar(255, 255, 255), 2);

			// Get all nonzero pixels within the window
			int x_sum = 0;
			for (auto p : nonzero) {
				if ((win_x_low < p.x) && (p.x < win_x_high) && (win_y_low < p.y) && (p.y < win_y_high)) {
					line_points.push_back(p);
					x_sum += p.x;
				}
			}

			// If there were enough white pixels in the window re-center
			if (line_points.size() >= MIN_NUMBER_PIXELS) {
				int x_mean = x_sum / line_points.size();
				x_current = x_mean;
				poly_points.insert(poly_points.end(), line_points.begin(), line_points.end());
			}
		}
		return poly_points;
	};

	std::vector<double> polyfit(std::vector<cv::Point> points, int degree = 2) {

		// Find length of points
		int N = points.size();
		int i, j, k;

		// Populate x and y
		std::vector<int> x(N), y(N);

		for (int i = 0; i < points.size(); ++i) {
			x[i] = points[i].y;
			y[i] = points[i].x;
		}

		//Array that will store the values of sigma(xi),sigma(xi^2),sigma(xi^3)....sigma(xi^2n)
		std::vector<int> X(2 * degree + 1);
		for (i = 0; i < 2 * degree + 1; i++)
		{
			X[i] = 0;
			for (j = 0; j < N; j++)
				//consecutive positions of the array will store N,sigma(xi),sigma(xi^2),sigma(xi^3)....sigma(xi^2n)
				X[i] = X[i] + pow(x[j], i);
		}

		//B is the Normal matrix(augmented) that will store the equations, 'a' is for value of the final coefficients
		std::vector<std::vector<double>> B(degree + 1, std::vector<double>(degree + 2));
		std::vector<double> a(degree + 1);

		for (i = 0; i <= degree; i++)
			for (j = 0; j <= degree; j++)
				//Build the Normal matrix by storing the corresponding coefficients at the right positions except the last column of the matrix
				B[i][j] = X[i + j];

		//Array to store the values of sigma(yi),sigma(xi*yi),sigma(xi^2*yi)...sigma(xi^n*yi)
		std::vector<int> Y(degree + 1);
		for (i = 0; i < degree + 1; i++)
		{
			Y[i] = 0;
			for (j = 0; j < N; j++)
				//consecutive positions will store sigma(yi),sigma(xi*yi),sigma(xi^2*yi)...sigma(xi^n*yi)
				Y[i] = Y[i] + pow(x[j], i) * y[j];
		}

		for (i = 0; i <= degree; i++)
			//load the values of Y as the last column of B(Normal Matrix but augmented)
			B[i][degree + 1] = Y[i];

		//n is made n+1 because the Gaussian Elimination part below was for n equations, but here n is the degree of polynomial and for n degree we get n+1 equations
		degree = degree + 1;
		for (i = 0; i < degree; i++)          
			//From now Gaussian Elimination starts(can be ignored) to solve the set of linear equations (Pivotisation)
			for (k = i + 1; k < degree; k++)
				if (B[i][i] < B[k][i])
					for (j = 0; j <= degree; j++)
					{
						double temp = B[i][j];
						B[i][j] = B[k][j];
						B[k][j] = temp;
					}

		for (i = 0; i < degree - 1; i++)            
			//loop to perform the gauss elimination
			for (k = i + 1; k < degree; k++)
			{
				double t = B[k][i] / B[i][i];
				for (j = 0; j <= degree; j++)
					//make the elements below the pivot elements equal to zero or elimnate the variables
					B[k][j] = B[k][j] - t * B[i][j];    
			}
		//back-substitution
		for (i = degree - 1; i >= 0; i--)                
		{                       
			//x is an array whose values correspond to the values of x,y,z..
			a[i] = B[i][degree];                
			//make the variable to be calculated equal to the rhs of the last equation
			for (j = 0; j < degree; j++)
				if (j != i)            
					//then subtract all the lhs values except the coefficient of the variable whose value                                   is being calculated
					a[i] = a[i] - B[i][j] * a[j];
			a[i] = a[i] / B[i][i];            
			//now finally divide the rhs by the coefficient of the variable to be calculated
		}

		/*
		std::cout << "\nThe values of the coefficients are as follows:\n";
		for (i = 0; i < degree; i++)
			std::cout << "x^" << i << "=" << a[i] << std::endl;           
		std::cout << "\nHence the fitted Polynomial is given by:\ny=";
		for (i = 0; i < degree; i++)
			std::cout << " + (" << a[i] << ")" << "x^" << i;
		std::cout << "\n";
		*/

		return a;

	};

public:

	std::vector<double> find_lane(cv::Mat& image, std::vector<cv::Point> nonzero, cv::Point current_base) {

		// Use sliding window to get points in line
		std::vector<cv::Point> poly_points = sliding_window(image, nonzero, current_base);

		// Fit polynomial to the line
		std::vector<double> coefficients = polyfit(poly_points);

		return coefficients;
	};
};