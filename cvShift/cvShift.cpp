// motion_correct.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"

#include <algorithm>

#include <opencv2/opencv.hpp>
#include <opencv2/opencv_lib.hpp>
#include <opencv2/matlab/bridge.hpp>
#pragma comment(lib, "libmx.lib")
#pragma comment(lib, "libmex.lib")
#pragma comment(lib, "libmat.lib")

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
	std::map<int, mxClassID> map_from_cv_to_mx;
	std::map<mxClassID, int> map_from_mx_to_cv;
	map_from_cv_to_mx[CV_64F] = mxDOUBLE_CLASS;
	map_from_cv_to_mx[CV_32F] = mxSINGLE_CLASS;
	map_from_cv_to_mx[CV_8S] = mxINT8_CLASS;
	map_from_cv_to_mx[CV_8U] = mxUINT8_CLASS;
	map_from_cv_to_mx[CV_16S] = mxINT16_CLASS;
	map_from_cv_to_mx[CV_16U] = mxUINT16_CLASS;
	map_from_cv_to_mx[CV_32S] = mxINT32_CLASS;
	for (auto i = map_from_cv_to_mx.begin(); i != map_from_cv_to_mx.end(); ++i)
		map_from_mx_to_cv[i->second] = i->first;

	try {
		std::vector<cv::Mat> source_images;
		int n_arg = 0;
		mwSize n_dimensions;
		const mwSize* p_size;
		std::vector<int> size;

		n_arg = 0;
		if (nrhs <= n_arg) mexErrMsgTxt("First argument is missing.");
		if (!mxIsNumeric(prhs[n_arg])) mexErrMsgTxt("First argument: must be numeric.");
		if (mxIsComplex(prhs[n_arg])) mexErrMsgTxt("First argument: must be real.");
		if (map_from_mx_to_cv.find(mxGetClassID(prhs[n_arg])) == map_from_mx_to_cv.end())
			mexErrMsgTxt("First argument: not convertible to OpenCV type.");
		n_dimensions = mxGetNumberOfDimensions(prhs[n_arg]);
		if (n_dimensions > 3) mexErrMsgTxt("First argument: larger than 3d array.");
		p_size = mxGetDimensions(prhs[n_arg]);
		int z = n_dimensions>2 ? p_size[2] : 1; 
		size = { static_cast<int>(p_size[1]), static_cast<int>(p_size[0])};
		source_images.reserve(z);
		for (int i = 0; i < z; i++) {
			source_images.push_back(cv::Mat(2, size.data(), map_from_mx_to_cv[mxGetClassID(prhs[n_arg])],
				static_cast<char*>(mxGetData(prhs[n_arg])) + i*size[0] * size[1] * mxGetElementSize(prhs[n_arg])));
		}
		
		n_arg = 1;
		if (nrhs <= n_arg) mexErrMsgTxt("Second argument is missing.");
		if (!mxIsNumeric(prhs[n_arg])) mexErrMsgTxt("Second argument: must be numeric.");
		if (mxIsComplex(prhs[n_arg])) mexErrMsgTxt("Second argument: must be real.");
		if (map_from_mx_to_cv.find(mxGetClassID(prhs[n_arg])) == map_from_mx_to_cv.end())
			mexErrMsgTxt("Second argument: not convertible to OpenCV type.");
		n_dimensions = mxGetNumberOfDimensions(prhs[n_arg]);
		if (n_dimensions > 2) mexErrMsgTxt("Second argument: larger than 2d array.");
		p_size = mxGetDimensions(prhs[n_arg]);
		if (p_size[0]!= source_images.size()) mexErrMsgTxt("Second argument: wrong size.");
		if (p_size[1]!=2) mexErrMsgTxt("Second argument: wrong size.");
		size = { static_cast<int>(p_size[1]), static_cast<int>(p_size[0]) };
		const cv::Mat input(2, size.data(), map_from_mx_to_cv[mxGetClassID(prhs[n_arg])],mxGetData(prhs[n_arg]));
		cv::Mat t;
		input.convertTo(t, CV_32F);

		std::vector<cv::Mat> results;
		results.reserve(source_images.size());
		cv::Mat tmp;
		for (int i = 0; i < source_images.size(); i++) {
			cv::Mat result;
			cv::Point2d center;
			center.x = source_images[i].cols / 2.0 - 0.5 - t.at<float>(i, 0);
			center.y = source_images[i].rows / 2.0 - 0.5 - t.at<float>(i, 1);
			source_images[i].convertTo(tmp, CV_32FC1);
			getRectSubPix(tmp, cv::Size(source_images[i].cols, source_images[i].rows), center, result);
			result.convertTo(result, source_images[i].depth());
			results.push_back(result);
		}

		std::vector<mwSize> output_size = { static_cast<mwSize>(results[0].size().width),
			static_cast<mwSize>(results[0].size().height), static_cast<mwSize>(results.size())};
		plhs[0] = mxCreateNumericArray(3, output_size.data(), map_from_cv_to_mx[results[0].depth()], mxREAL);
		uchar* p_data = static_cast<uchar*> (mxGetData(plhs[0]));
		int plane_size = output_size[0] * output_size[1] * mxGetElementSize(plhs[0]);
		for (int i = 0; i<results.size(); i++) {
			std::copy(results[i].data, results[i].data + plane_size, p_data + i * plane_size);
		}
		return;
	}
	catch (std::exception & e) {
		mexErrMsgTxt(e.what());
		return;
	}
}