#include "mex.h"
#include "mmap.h"
#include <map>
#include <string>
#include <memory>
#pragma comment(lib, "libmx.lib")
#pragma comment(lib, "libmex.lib")
#pragma comment(lib, "libmat.lib")

static std::map<std::string, std::shared_ptr<MMap<SmallDoubleMatrix>>> map_mmap;

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
	map_mmap[basename] = std::make_shared<MMap<SmallDoubleMatrix>>(basename);
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
				if (mxGetClassID(prhs[2])!= mxDOUBLE_CLASS) exit_with_error("Third argument should be double.");
				mwSize numel = mxGetNumberOfElements(prhs[2]);
				if (numel > SmallDoubleMatrix::max_size) exit_with_error("Too large third matrix. Max number of elements = " + std::to_string(SmallDoubleMatrix::max_size));
				mwSize n_dimensions = mxGetNumberOfDimensions(prhs[2]);
				if (n_dimensions>2) exit_with_error("Third argument should be 2D.");

				SmallDoubleMatrix data;
				const mwSize* p_size = mxGetDimensions(prhs[2]);
				data.height = p_size[0];
				data.width = p_size[1];
				double *pr = mxGetPr(prhs[2]);
				if (!pr) exit_with_error("No real data in the third argument.");

				std::copy(pr, pr + numel, data.data);
				int res = map_mmap[basename]->set(data, sizeof(data) - sizeof(data.data) + sizeof(double)*numel)<0;
				if (res<0) exit_with_error("Failed to write to mmap. Error Code: " + std::to_string(-res));
			}
			else {
				exit_with_error("Double Matrix required.");
			}
		}
		else if (cmd == "get") {
			initialize(basename);
			SmallDoubleMatrix data;
			int res = map_mmap[basename]->get(data);
			if (res<0) exit_with_error("Failed to read mmap. Error Code: " + std::to_string(-res));
			mwSize return_size[] = { data.height,data.width };
			plhs[0] = mxCreateNumericArray(2, return_size, mxDOUBLE_CLASS, mxREAL);
			std::copy(data.data, data.data+data.height*data.width, mxGetPr(plhs[0]));
		}
		else {
			exit_with_error("Unknown command.");
		}
	}
	print_mmap_error();
}




