#include "mex.h"
#include "mmap.h"
#include <map>
#include <string>
#include <memory>
#pragma comment(lib, "libmx.lib")
#pragma comment(lib, "libmex.lib")
#pragma comment(lib, "libmat.lib")

static std::map<std::string, std::shared_ptr<MMap<SI4Image>>> map_mmap;

void print_mmap_error() {
	for (auto it = map_mmap.begin(); it != map_mmap.end(); ++it) {
		if (it->second) {
			std::string error = it->second->get_error();
			if (!error.empty()) mexWarnMsgTxt(error.c_str());
		}
	}
}

void exit_with_error(std::string error_string) {
	print_mmap_error();
	mexErrMsgTxt(error_string.c_str());
}

void clear(void) {
	map_mmap.clear();
	if (mexIsLocked()) mexUnlock();
}

void initialize(std::string basename) {
	if (map_mmap.find(basename) != map_mmap.end()) return;
	map_mmap[basename] = std::make_shared<MMap<SI4Image>>(basename);
	if (map_mmap[basename]->is_valid()) {
		if (!mexIsLocked()) mexLock();
	}
	else {
		std::string error_str = map_mmap[basename]->get_error();
		map_mmap.erase(basename);
		exit_with_error("Error occured during MMap initialization. " + error_str);
	}
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
	if (nrhs<1) exit_with_error("Require first argument (command).");
	if (mxGetClassID(prhs[0]) != mxCHAR_CLASS) exit_with_error("First argument should be a command string.");
	char * p_cmd = mxArrayToString(prhs[0]);
	std::string cmd(p_cmd);
	mxFree(p_cmd);

	if (cmd == "clear") {
		clear();
	}
	else {
		if (nrhs<2 || mxGetClassID(prhs[1]) != mxCHAR_CLASS) exit_with_error("Require second argument (base name).");
		char * p_basename = mxArrayToString(prhs[1]);
		std::string basename = p_basename;
		mxFree(p_basename);

		if (cmd == "is_valid") {
			plhs[0] = mxCreateLogicalScalar(map_mmap.find(basename) != map_mmap.end() && map_mmap[basename]->is_valid());
		}
		else if (cmd == "set") {
			initialize(basename);
			if (nrhs>2) {
				SI4Image image;
				if (mxGetClassID(prhs[2])!= mxINT16_CLASS) exit_with_error("Third argument should be 512*512 or 512*512*2 INT16.");
				mwSize ndim = mxGetNumberOfDimensions(prhs[2]);
				mwSize numel = mxGetNumberOfElements(prhs[2]);
				const mwSize* sizes = mxGetDimensions(prhs[2]);
				if (ndim > 3) exit_with_error("Third argument should have 2 or 3 dims.");
				if (numel > image.max_size) exit_with_error("Third argument too large");
				image.height = sizes[0];
				image.width = sizes[1];
				image.n_ch = ndim == 3 ? sizes[2] : 1;
				void *pr = mxGetData(prhs[2]);
				if (!pr) exit_with_error("No real data in the third argument.");
				memcpy((void*)image.data, pr, sizeof(int16_t)*image.height*image.width* image.n_ch);
				if (nrhs > 3) {
					if (mxGetClassID(prhs[3]) != mxDOUBLE_CLASS || mxGetNumberOfElements(prhs[3])!=1) exit_with_error("Forth argument should be a Double scalar.");
					double* tmp = mxGetPr(prhs[3]);
					if (!tmp) exit_with_error("No real data in the forth argument");
					image.frame_tag = *tmp;
				}else{
					image.frame_tag = 0;
				}
				int res = map_mmap[basename]->set(image);
				if (res<0) exit_with_error("Failed to write to mmap. Error Code: " + std::to_string(-res));
			}
			else {
				exit_with_error("Double Matrix required.");
			}
		}
		else if (cmd == "get") {
			initialize(basename);
			SI4Image image;
			int res = map_mmap[basename]->get(image);
			if (res < 0) exit_with_error("Failed to read mmap. Error Code: " + std::to_string(-res));
			if (res > 0) {
				mxAssert(image.height * image.width * image.n_ch <= image.max_size, "Matrix Size Corrupted");
				mwSize return_size[] = { image.height, image.width, image.n_ch};
				plhs[0] = mxCreateNumericArray(3, return_size, mxINT16_CLASS, mxREAL);
				memcpy(mxGetData(plhs[0]), (void*)image.data, sizeof(int16_t)*image.height*image.width*image.n_ch);
				if (nlhs > 1) plhs[1] = mxCreateDoubleScalar(image.frame_tag);
			}
			else {
				mwSize return_size[] = { 0, 0 };
				plhs[0] = mxCreateNumericArray(2, return_size, mxINT16_CLASS, mxREAL);
				if (nlhs > 1) plhs[1] = mxCreateDoubleScalar(0);
			}
		}
		else {
			exit_with_error("Unknown command.");
		}
	}
	print_mmap_error();
}




