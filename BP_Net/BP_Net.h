#pragma once
#include <Eigen>
#include <Dense>
#include <vector>

class CBP_Net
{
	using Matrixfd = Eigen::Matrix<float, -1, -1, Eigen::ColMajor>;
	using vec_int = std::vector<int>;
	using vec_float = std::vector<float>;

public:
	CBP_Net();
	~CBP_Net();

	// 设置各层神经元节点数
	void setLayers(vec_int& nodes_num);
	// 训练BP神经网络模型
	void train(const std::vector<vec_float>& train_x, const vec_int& train_y, const size_t loops=10);
	// 依据训练好的模型进行分类
	int predict(const vec_float test_x);

private:
	// 编码转换: 将标签多列化
	void labelEncoder(const vec_int& train_y, std::vector<Matrixfd>& Y);
	// 利用每一组{属性，标签}
	float process(Matrixfd& x, Matrixfd& y);
	// 前向传播
	void  feed_forward(Matrixfd& x);
	// 误差反向传播
	float  feed_backward(Matrixfd& x, Matrixfd& y);

	Matrixfd m_w1, m_w2, m_b1, m_b2, m_middle, m_midone, m_FI, m_output, m_err;
	Matrixfd m_dw2, m_w2e, m_dw1;
	int m_n{ 5 }, m_l{ 3 }, m_m{ 2 };
	float m_rate{ 0.1f };

	std::vector<std::pair<int, Matrixfd>>  m_labelEncoder;
};
