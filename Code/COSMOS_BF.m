clc
clear all
close all

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

load(['data\rec_train_',int2str(NChan),'_DATASET8.mat']);
load('data\DATASET8.mat');


DAS_Nchan = uint8(IQ2Bmode(hilbert(squeeze(sum(images.data(2,:,channels,:),3)))));
Deep_Nchan = uint8(IQ2Bmode(squeeze(rec(1,:,1,:)) + sqrt(-1)*squeeze(rec(1,:,2,:))));

figure
subplot(1,2,1), imagesc(DAS_Nchan), colormap gray
title(['DAS-',int2str(NChan)])
subplot(1,2,2), imagesc(Deep_Nchan), colormap gray
title(['DeepBF-',int2str(NChan)])

imwrite(uint8(DAS_Nchan),['Results\DAS_',int2str(NChan),'.bmp'],'bmp');
imwrite(uint8(Deep_Nchan),['Results\DeepBF_',int2str(NChan),'.bmp'],'bmp');

DAS_Ref=imread(['Results\DAS_64.bmp']);
DeepBF_Ref=imread(['Results\DeepBF_64.bmp']);

[psnr(DAS_Ref,DAS_Nchan) psnr(DeepBF_Ref,Deep_Nchan) ssim(DAS_Ref,DAS_Nchan,'Radius',50) ssim(DeepBF_Ref,Deep_Nchan,'Radius',50)]

end
