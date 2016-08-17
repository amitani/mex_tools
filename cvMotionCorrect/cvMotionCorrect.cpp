// motion_correct.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "image_registrator.h"

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
		if(nrhs <= n_arg) mexErrMsgTxt("First argument is missing.");
		if (!mxIsNumeric(prhs[n_arg])) mexErrMsgTxt("First argument: must be numeric.");
		if (mxIsComplex(prhs[n_arg])) mexErrMsgTxt("First argument: must be real.");
		if (map_from_mx_to_cv.find(mxGetClassID(prhs[n_arg])) == map_from_mx_to_cv.end())
			mexErrMsgTxt("First argument: not convertible to OpenCV type.");
		n_dimensions = mxGetNumberOfDimensions(prhs[n_arg]);
		if (n_dimensions > 3) mexErrMsgTxt("First argument: larger than 3d array.");
		p_size = mxGetDimensions(prhs[n_arg]);
		int z = n_dimensions>2 ? p_size[2] : 1; 
		size = { static_cast<int>(p_size[0]), static_cast<int>(p_size[1]) };
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
		size = { static_cast<int>(p_size[0]), static_cast<int>(p_size[1]) };
		cv::Mat template_image(2, size.data(), map_from_mx_to_cv[mxGetClassID(prhs[n_arg])],mxGetData(prhs[n_arg]));

		matlab::ArgumentParser parser("");
		parser.addVariant("align", 0, 6, std::vector<std::string>{"factor", "margin",
			"sigma_smoothing", "sigma_normalization", "normalization_offset", "to_equalize_histogram"});
		matlab::MxArrayVector raw(prhs + 2, prhs + nrhs);
		matlab::MxArrayVector reordered = parser.parse(raw);
		cv::bridge::BridgeVector inputs(reordered.begin(), reordered.end());
		double factor = inputs[0].empty() ? -1 : inputs[0].toDouble();
		int    margin = inputs[1].empty() ? -1 : inputs[1].toDouble();
		double sigma_smoothing = inputs[2].empty() ? -1 : inputs[2].toDouble();
		double sigma_normalization = inputs[3].empty() ? -1 : inputs[3].toDouble();
		double normalization_offset = inputs[4].empty() ? -1 : inputs[4].toDouble();
		int to_equalize_histogram = inputs[5].empty() ? -1 : inputs[5].toDouble();

		ImageRegistrator image_registrator;
		image_registrator.SetTemplate(template_image);
		image_registrator.SetParameters(factor, margin, sigma_smoothing, sigma_normalization,
			normalization_offset, to_equalize_histogram);
		image_registrator.Init();
		std::vector<cv::Point2d> results;
		results.reserve(source_images.size());
		std::vector<cv::Mat> heatmaps;
		heatmaps.reserve(source_images.size());
		for (int i = 0; i < source_images.size(); i++) {
			cv::Point2d result;
			cv::Mat heatmap;
			image_registrator.Align(source_images[i], NULL, &result, &heatmap);
			results.push_back(result);
			heatmaps.push_back(heatmap);
		}

		plhs[0] = mxCreateNumericMatrix(source_images.size(), 2, mxDOUBLE_CLASS, mxREAL);
		double * rv = (double *)(mxGetData(plhs[0]));
		for (int i = 0; i<z; i++) {
			rv[i + 0 * z] = -results[i].x;
			rv[i + 1 * z] = -results[i].y;
		}
		if(nlhs>1){
			std::vector<mwSize> output_size = { static_cast<mwSize>(heatmaps[0].size().width),
				static_cast<mwSize>(heatmaps[0].size().height), static_cast<mwSize>(heatmaps.size()) };
			plhs[1] = mxCreateNumericArray(3, output_size.data(), map_from_cv_to_mx[heatmaps[0].depth()], mxREAL);
			uchar* p_data = static_cast<uchar*> (mxGetData(plhs[1]));
			int plane_size = output_size[0] * output_size[1] * mxGetElementSize(plhs[1]);
			for (int i = 0; i<heatmaps.size(); i++) {
				std::copy(heatmaps[i].data, heatmaps[i].data + plane_size, p_data + i * plane_size);
			}
		}

		//plhs[0] = cv::bridge::Bridge::FromMat<double>(cv::Mat(results)).releaseOwnership();
		//plhs[0] = cv::bridge::Bridge::FromMat<double>(heatmaps[0]).releaseOwnership();
		//plhs[0] = cv::bridge::Bridge::FromMat<double>(cv::Mat(source_images[0])).releaseOwnership();
		//plhs[0] = cv::bridge::Bridge::FromMat<double>(source_images_mat).releaseOwnership();
		//plhs[0] = mxCreateDoubleScalar(margin);
		return;
	}
	catch (cv::Exception e) {
		mexErrMsgTxt(e.what());
		return;
	}
}