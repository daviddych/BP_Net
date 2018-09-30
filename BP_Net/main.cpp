#include <iostream>
#include "BP_Net.h"
#include "LoadFile.h"

using namespace Eigen;

int main(int argc, char** argv)
{
	// load data
	CLoadFile txtfile;
	const char* filename = R"(data.txt)"; 
	std::vector<std::vector<float>> data_x;
	std::vector<int> data_y;
	txtfile.Read(filename, data_x, data_y);

	for (int  loop = 10; loop < 1000; loop += int(loop * 0.1))
	{
		// train BP-Net Model
		CBP_Net bp_net;
		std::vector<int> nodes_num{ 24,25,4 };
		bp_net.setLayers(nodes_num);
		bp_net.train(data_x, data_y, loop);

		// train precision.
		int rights[4] = { 0 };
		int total[4] = { 0 };
		for (size_t i = 0; i < data_x.size(); ++i)
		{
			++total[data_y[i] - 1];
			if (data_y[i] == bp_net.predict(data_x[i]))
				++rights[data_y[i] - 1];
		}

		printf("============================%d===========================\n", loop);
		for (size_t i = 0; i < 4; ++i)
		{
			printf("%d/%d = %f\n", rights[i], total[i], rights[i] * 1.0f / total[i]);
		}
		printf("\n");
	}
	

	return 0;
}