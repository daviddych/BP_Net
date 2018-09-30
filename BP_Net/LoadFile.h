#pragma once
#include <vector>

class CLoadFile
{
public:
	static bool Read(const char* filename, std::vector<std::vector<float>>& data_x, std::vector<int>& data_y);
	CLoadFile();
	~CLoadFile();
};

