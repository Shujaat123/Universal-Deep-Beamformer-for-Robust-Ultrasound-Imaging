# Universal-Deep-Beamformer-for-Robust-Ultrasound-Imaging
Computer code and dataset for "Universal Deep Beamformer for Robust Ultrasound Imaging"

Paper
===============
* Shujaat Khan, Jaeyoung Huh, and Jong Chul Ye. "Universal Deep Beamformer for Variable Rate Ultrasound Imaging." https://arxiv.org/abs/1901.01706.

Implementation
===============
* MatConvNet (matconvnet-1.0-beta24)
  * Please run the matconvnet-1.0-beta24/matlab/vl_compilenn.m file to compile matconvnet.
  * There is instruction on "http://www.vlfeat.org/matconvnet/mfiles/vl_compilenn/"
  * Please run the installation setup (install.m) and run some training examples.
 
Trained network
===============
* Trained network for 'Universal Deep Beamformer CNN' is uploaded.
* Use 100th or 200th epoch weights for validation. (Results in paper were generated using 200th epoch's weights)

Test data
===============
* A sample test data is placed in 'data\' folder.
* The dimensions of data are as follows
  -- Test_data      =  3x96x64x2048  (input-planes x scanlines x channels x depth)
                        
To perform a test using proposed algorithm

-> Run 'DeepBF_Test.m
