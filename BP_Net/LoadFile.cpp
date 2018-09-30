#include "LoadFile.h"
#include <assert.h>
#include <fstream>
#include <iostream>
#include <string>
#include <iosfwd>
#include <sstream>

CLoadFile::CLoadFile()
{
}

CLoadFile::~CLoadFile()
{
}

bool CLoadFile::Read(const char* filename, std::vector<std::vector<float>>& data_x, std::vector<int>& data_y)
{
	assert(filename != NULL);
	std::ifstream infile(filename);
	if (!infile.is_open())
	{
#ifdef _CONSOLE
		std::cout << "Failed open file: " << std::string(filename) << std::endl;
#endif
		return false;
	}

	float label;
	while (!infile.eof())
	{
		infile >> label;
		data_y.push_back(static_cast<int>(label));
		
		std::vector<float> sub_data(24);
		for (size_t i = 0; i < 24; ++i)
		{
			infile >> sub_data[i];
		}
		data_x.push_back(sub_data);
	}

	return true;
}