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
		size = { static_cast<int>(p_size[1]), static_cast<int>(p_size[0]) };
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
		size = { static_cast<int>(p_size[1]), static_cast<int>(p_size[0]) };
		cv::Mat template_image(2, size.data(), map_from_mx_to_cv[mxGetClassID(prhs[n_arg])], mxGetData(prhs[n_arg]));


		n_arg = 2;
		if (nrhs <= n_arg) mexErrMsgTxt("Third argument is missing.");
		if (!mxIsNumeric(prhs[n_arg])) mexErrMsgTxt("Third argument: must be numeric.");
		if (mxIsComplex(prhs[n_arg])) mexErrMsgTxt("Third argument: must be real.");
		if (map_from_mx_to_cv.find(mxGetClassID(prhs[n_arg])) == map_from_mx_to_cv.end())
			mexErrMsgTxt("Third argument: not convertible to OpenCV type.");
		n_dimensions = mxGetNumberOfDimensions(prhs[n_arg]);
		if (n_dimensions > 2) mexErrMsgTxt("Third argument: larger than 2d array.");
		p_size = mxGetDimensions(prhs[n_arg]);
		if (p_size[0]<2 || p_size[0] > 3) mexErrMsgTxt("Third argument: wrong size.");
		if (p_size[1]<3 || p_size[1] > 3) mexErrMsgTxt("Third argument: wrong size.");
		size = { static_cast<int>(p_size[1]), static_cast<int>(p_size[0]) };
		cv::Mat tmp(2, size.data(), map_from_mx_to_cv[mxGetClassID(prhs[n_arg])], mxGetData(prhs[n_arg]));
		tmp = tmp.t();
		cv::Mat warp(tmp.clone());
		tmp.row(1).copyTo(warp.row(0));
		tmp.row(0).copyTo(warp.row(1));
		warp.copyTo(tmp);
		tmp.col(1).copyTo(warp.col(0));
		tmp.col(0).copyTo(warp.col(1));  // Matlab is column-based and cv::Mat is row-based

		n_arg = 3;
		if (nrhs <= n_arg) mexErrMsgTxt("Fourth argument is missing.");
		if (!mxIsNumeric(prhs[n_arg])) mexErrMsgTxt("Fourth argument: must be numeric.");
		if (mxIsComplex(prhs[n_arg])) mexErrMsgTxt("Fourth argument: must be real.");
		if (map_from_mx_to_cv.find(mxGetClassID(prhs[n_arg])) == map_from_mx_to_cv.end())
			mexErrMsgTxt("Fourth argument: not convertible to OpenCV type.");
		n_dimensions = mxGetNumberOfDimensions(prhs[n_arg]);
		if (n_dimensions > 2) mexErrMsgTxt("Fourth argument: larger than 2d array.");
		p_size = mxGetDimensions(prhs[n_arg]);
		size = { static_cast<int>(p_size[1]), static_cast<int>(p_size[0]) };
		cv::Mat mask(2, size.data(), map_from_mx_to_cv[mxGetClassID(prhs[n_arg])], mxGetData(prhs[n_arg]));
		mask.convertTo(mask, CV_8U);

		/*
		n_arg = 2;
		bool inv = true;
		if (nrhs > n_arg) {
			if (!mxIsChar(prhs[n_arg])) mexErrMsgTxt("Third argument: must be numeric.");
			char * p_str = mxArrayToString(prhs[n_arg]);
			std::string opt(p_str);
			mxFree(p_str);
			if (!opt.empty()) {
				if (opt == "inv")
					inv = false;
				else
					mexErrMsgTxt("Third argument: unknown option (not 'inv').");
			}
		}
		*/

		std::vector<cv::Mat> results;// = source_images;
									 //results.push_back(warp);
		results.reserve(source_images.size());
		for (int i = 0; i < source_images.size(); i++) {
			cv::Mat result;
			warp.convertTo(result, CV_32F);
			try {
				cv::findTransformECC(template_image, source_images[i], result,cv::MOTION_AFFINE,
					cv::TermCriteria(cv::TermCriteria::COUNT+cv::TermCriteria::EPS,50,0.001),mask);
			}
			catch (cv::Exception e) {
				mexWarnMsgTxt(e.what());
			}
			results.push_back(result);
		}

		std::vector<mwSize> output_size = { static_cast<mwSize>(results[0].size().width),
			static_cast<mwSize>(results[0].size().height), static_cast<mwSize>(results.size()) };
		plhs[0] = mxCreateNumericArray(3, output_size.data(), map_from_cv_to_mx[results[0].depth()], mxREAL);
		uchar* p_data = static_cast<uchar*> (mxGetData(plhs[0]));
		int plane_size = output_size[0] * output_size[1] * mxGetElementSize(plhs[0]);
		//std::vector<uchar> test{ 0,10,00,20,00,30 };
		for (int i = 0; i<results.size(); i++) {
			//std::copy(test.begin(), test.end(), p_data + i * plane_size);
			std::copy(results[i].data, results[i].data + plane_size, p_data + i * plane_size);
		}
		//*reinterpret_cast<int16_t*>(p_data) = results.size();
		//*(reinterpret_cast<int16_t*>(p_data) + 1) = results[0].size().height;
		//*(reinterpret_cast<int16_t*>(p_data) + 2) = results[0].size().width;
		//*(reinterpret_cast<int16_t*>(p_data) + 3) = output_size[0];
		//*(reinterpret_cast<int16_t*>(p_data) + 4) = output_size[1];
		//*(reinterpret_cast<int16_t*>(p_data) + 5) = output_size[2];
		//*(reinterpret_cast<int16_t*>(p_data) + 5) = mxGetElementSize(plhs[0]);



		//plhs[0] = cv::bridge::Bridge::FromMat<double>(cv::Mat(results)).releaseOwnership();
		//plhs[0] = cv::bridge::Bridge::FromMat<double>(heatmaps[0]).releaseOwnership();
		//plhs[0] = cv::bridge::Bridge::FromMat<double>(cv::Mat(source_images[0])).releaseOwnership();
		//plhs[0] = cv::bridge::Bridge::FromMat<double>(source_images_mat).releaseOwnership();
		//plhs[0] = mxCreateDoubleScalar(margin);
		return;
	}
	catch (std::exception & e) {
		mexErrMsgTxt(e.what());
		return;
	}
}