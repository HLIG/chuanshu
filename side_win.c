#include<stdio.h>
#include<stdlib.h>
typedef int int32_t;
typedef unsigned short uint16_t;
void cal_grad_x(const float* pusInputImage, float* pusOutputGradx, int32_t nHeight, int32_t nWidth, int32_t nWin,float diff_thres)
{
	--nWin;
	int32_t 
	for (int32_t row = 0; row < nHeight; ++row) {
		float *col_data = pusInputImage + row * nWidth;
		for (int32_t col = 0; col < nWidth; ++col) {

		}
	}
}

vector<int> maxSlidingWindow(vector<int>& nums, int k) {
	k--;
	vector<int> res;
	if (nums.empty()) return res;

	deque<int> d;  //存放元素的下标
	int n = nums.size();
	for (int i = 0; i < n; i++) {
		//先检查队首是否是不在窗口中（front<i-k）,不在的话就将其弹出；
		if (!d.empty() && d.front() < i - k) {
			d.pop_front();
		}

		//再根据要加入队列的元素nums[i]是否大于队列尾(即目前除最大值以外的下标)，将之前的数都清除，保持队列单调递减
		while (!d.empty() && nums[d.back()] < nums[i]) {
			d.pop_back();
		}
		d.push_back(i);

		//如果元素滑到了第一个窗口大小右方，那就开始加入最大值(nums[d.front()])
		if (i >= k)
		{
			res.push_back(nums[d.front()]);
		}
	}

	return res;


}