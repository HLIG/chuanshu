int BinarySearch(float* nums, float target, int left, int right)
{
	while (left < right-1){
		int mid = left + ((right - left) >> 1);   
		(target < nums[mid]) ? right = mid-1 : left = mid ;  
	}
	return left; 
}
void rowNLM(float *pfimage, const float *pfGuideImage, int32_t nHeight, int32_t nWidth, int32_t KernelSize,int32_t PatchSize,int32_t step) {
	int32_t kRadius = KernelSize >> 2;
	int32_t pRadius = PatchSize >> 2;
	for (int32_t row = 0; row < nHeight; ++row) {
		for (int32_t col = kRadius; col < nWidth - kRadius; ++col) {
			int32_t CenterIndex = row * nWidth + col;
			float sumValue = pfimage[CenterIndex];
			float sumWeight = 1.;
			for(int32_t pRadius)
		}
	}
}
