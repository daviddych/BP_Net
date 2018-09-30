#include "BP_Net.h"
#include <iostream>

CBP_Net::CBP_Net(){
}

CBP_Net::~CBP_Net(){
}

void CBP_Net::setLayers(vec_int& nodes_num) { // 设置各层神经元节点数
	assert(nodes_num.size() == 3);

	m_n = nodes_num[0];
	m_l = nodes_num[1];
	m_m = nodes_num[2];

	m_w1.setRandom(m_l, m_n); m_w1 = m_w1.cwiseAbs();
	m_dw1.setRandom(m_l, m_n);
	m_w2.setRandom(m_m, m_l); m_w2 = m_w2.cwiseAbs();
	m_dw2.setRandom(m_m, m_l);
	m_w2e.setRandom(m_m, 1);
	m_b1.setRandom(m_l, 1);
	m_b2.setOnes(m_m, 1);
	m_FI.resize(m_m, 1);
	m_middle.resize(m_l, 1);
	m_output.resize(m_m, 1);
	m_err.resize(m_m, 1);
	m_midone.setOnes(m_l, 1);
}

void CBP_Net::labelEncoder(const vec_int& train_y, std::vector<Matrixfd>& Y)
{
	const size_t  N = train_y.size();

	// 将标签多列化
	vec_int label(N);
	std::copy(train_y.begin(), train_y.end(), label.begin());
	std::sort(label.begin(), label.end());
	auto end_iter = std::unique(label.begin(), label.end());
	label.erase(end_iter, std::end(label));
	Y.resize(N);

	const size_t  M = label.size();
	size_t i, j;
	for (i = 0; i < N; ++i) {
		Y[i].setZero(M, 1);

		for (j = 0; j < M; ++j) {
			if (label[j] == train_y[i]) {
				Y[i](j) = 1;
			}
		}
	}

	for (j = 0; j < M; ++j)
	{
		Matrixfd en;
		en.setZero(M, 1);
		en(j) = 1;
		m_labelEncoder.push_back(std::make_pair(label[j], en));
	}
}

void CBP_Net::train(const std::vector<vec_float>& train_x,
	const vec_int& train_y, const size_t loops) { // train BP-Net model
	assert(!train_x.empty() && !train_y.empty() && train_x.size() == train_y.size());

	std::vector<Matrixfd> Y;
	labelEncoder(train_y, Y);

	assert((Y[0].rows() == m_m && Y[0].cols() == 1) || (Y[0].cols() == m_m && Y[0].rows() == 1));

	const size_t  N = train_y.size();   
	const size_t  M = train_x[0].size(); 

	Matrixfd  x;
	x.resize(M, 1);
	size_t i, j, loop;
	for (loop = 0; loop < 50; ++loop) {
		for (i = 0; i < N; ++i)		{
			Matrixfd& y = Y[i];
			const vec_float& tx = train_x[i];
			for (j = 0; j < M; ++j)
				x(j) = tx[j];

			process(x, y);
		}
	}
}

float  CBP_Net::feed_backward(Matrixfd& x, Matrixfd& y)
{
	// error between target and output.
	m_err = y - m_output; 

	// update the with and threshold
	m_dw2 = m_err * m_middle.transpose(); 

	m_FI = m_middle.cwiseProduct(m_midone - m_middle);
	m_w2e = m_w2.transpose() * m_err;
	for (size_t k = 0; k < m_n; ++k)
	{
		for (size_t j = 0; j < m_l; ++j)
			m_dw1(j, k) = m_FI(j) * x(k) * m_w2e(j);
	}

	m_w1 = m_w1 + m_rate * m_dw1;
	m_b1 = m_b1 + m_rate * (m_FI.cwiseProduct(m_w2e));
	m_w2 = m_w2 + m_rate * m_dw2;
	m_b2 = m_b2 + m_rate * m_err;

	return m_err.cwiseAbs().sum();
}

float CBP_Net::process(Matrixfd& x, Matrixfd& y) { 
	feed_forward(x);
	float e = feed_backward(x, y);

// 	std::cout << m_output.transpose() << " ==== " << y.transpose() << "\t"
// 		<< "t_e = " << e << std::endl;
	return e;
}

void  CBP_Net::feed_forward(Matrixfd& x)
{
	size_t i{ 0 };

	// middle hide layer
	m_middle = m_w1 * x + m_b1;
	for (i = 0; i < m_l; ++i)
		m_middle(i) = 1.0f / (1.0f + exp(-m_middle(i))); // sigmoid

	// output lay
	m_output = m_w2 * m_middle + m_b2;
}

int CBP_Net::predict(const vec_float test_x)
{
	Matrixfd  x;
	x.resize(test_x.size(), 1);
	for (size_t j = 0; j < test_x.size(); ++j)
		x(j) = test_x[j];

	feed_forward(x);

	//m_output = m_output.cwiseAbs();
	//size_t idx = { 0 };
	//float min_v = m_output(0);
	//for (size_t i = 1; i < m_output.size(); ++i)
	//{
	//	if (m_output(i) > min_v)
	//	{
	//		min_v = m_output(i);
	//		idx = i;
	//	}
	//}
	//return m_labelEncoder[idx].first;

	size_t i{ 0 }, j{ 0 }, k{ 0 };
	float min_dist = std::numeric_limits<float>::max(), dist;
	int re_cls;
	for (i = 0; i < m_labelEncoder.size(); ++i)
	{
		std::pair<int, Matrixfd>& le = m_labelEncoder[i];
		dist = (m_output - le.second).norm();		
		if (dist < min_dist)
		{
			min_dist = dist;
			re_cls = le.first;
		}
	}

	return re_cls;
}