%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Made by 'Shujaat Khan' (shujaat@kaist.ac.kr) at 2019.01.12
% Paper : Universal Deep Beamformer for Variable Rate Ultrasound Imaging
%         https://arxiv.org/abs/1901.01706

% Copyright <2018> <Shujaat Khan' (shujaat@kaist.ac.kr)>
% 
% Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
% 
% 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

% 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
% 
% 3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
% 
% THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" 
% AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, 
% THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. 
% IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, 
% OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; 
% LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, 
% WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, 
% EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%

clear;
reset(gpuDevice(1));
restoredefaultpath();

addpath('../lib');

run('matlab/vl_setupnn');

rng('shuffle');

%%
epoch       	= 100;

%%
wgt             = 1e3;
inputSize    	= [64, 96, 3];
wrapSize        = [0, 0, 1];

%%
method_        	= 'normal';
layerFile     	= @cnn_sparse_view_init_multi;

expDir          = ['data/' func2str(layerFile) '_' method_];
addDir          = ['_input' num2str(inputSize(1))];

expDirRM4896    = [expDir, addDir];
%%
netRM4896             = load([expDirRM4896, '/net-epoch-' num2str(epoch) '.mat']);
gpus                = 2;
mode                = 'test';
dataDir             = [expDirRM4896];
dataFile            = ['DATASET8.mat'];

% Evaluation on variable down-sampling ratios
for NChan = [64, 32, 24, 16, 8, 4]
 if NChan==64
    channels = [randperm(31,31) 32, 33, 33+randperm(31,31)];
elseif NChan==32
    channels = [randperm(31,15) 32, 33, 33+randperm(31,15)];
elseif NChan==24
    channels = [randperm(31,11) 32, 33, 33+randperm(31,11)];
elseif NChan==16
    channels = [randperm(31,7) 32, 33, 33+randperm(31,7)];
elseif NChan==8
    channels = [randperm(31,3) 32, 33, 33+randperm(31,3)];
elseif NChan==4
    channels = [randperm(31,1) 32, 33, 33+randperm(31,1)];
end
        
imdb                = load(fullfile(['data\',dataFile]));
temp = imdb.images.data;
imdb.images.data = 0.*imdb.images.data;
imdb.images.data(:,:,channels,:) = temp(:,:,channels,:);
imdb.images.data=permute(imdb.images.data,[3 2 1 4]);
imdb.images.data = single(imdb.images.data);


%%
opts.inputSize      = [size(imdb.images.data,1), size(imdb.images.data,2), size(imdb.images.data,3)];
opts.wrapSize       = wrapSize;

nZ                 = size(imdb.images.data,4);


opt_                = netRM4896.net.meta.trainOpts;
net_                = netRM4896.net;
net_.layers(end)    = [];

            if gpus
                net_     = vl_simplenn_move(net_, 'gpu') ;
            end

for iz = 1:nZ
                
    if gpus
        data_   = (imdb.images.data(:,:,:,iz) * wgt);
        data_   = gpuArray(data_);
    else
        data_   = (imdb.images.data(:,:,:,iz) * wgt);        
    end

    res_        = vl_simplenn(net_, data_, [], [], 'conserveMemory', 0, 'mode', mode, 'accumulate', 0, ...
                                                    'backPropDepth', inf, 'sync', 0, 'cudnn', 1);

    rec_                = gather(res_(end).x ./ wgt);
    data_               = gather(data_ ./ wgt);

    clear res_;
    rec(:,:,:,iz)	= rec_;
    disp([num2str(iz) ' / ' num2str(nZ)]);
            
end

save(['data/rec_train_',int2str(NChan),'_',dataFile], 'rec', '-v7.3');
end        

COSMOS_BF
