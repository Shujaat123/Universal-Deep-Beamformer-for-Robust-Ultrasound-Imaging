clc
clear;
reset(gpuDevice);

rng('shuffle');

dataDir             = ['data'];

dataFile            = ['Multi_CH_vDSR.mat'];

imdb                = load(fullfile(dataDir, dataFile));

imdb.images.labels=permute(imdb.images.labels,[3 2 1 4]);

%% Data Parameters
opts.dataDir = dataDir;
opts.dataFile = dataFile;
opts.batchNormalization = false ;
opts.networkType = 'simplenn' ;
opts.method         = 'normal';
opts.inputRange     = [-1, 1];
opts.inputSize      = [64, 96, 3];
opts.wrapSize       = [0, 0, 1];

opts.cwgt           = 1e0;
opts.grdclp         = [-1e-2, 1e-2];
opts.lrnrate     	= [-4, -5];
opts.wgtdecay     	= 1e-4;
 
opts.lv             = [];
opts.pflt           = 'pyr';
opts.dflt           = 'cd';

opts.wgt            = 1e3;
opts.offset         = 0;
 
opts.numEpochs      = 100;
opts.batchSize      = 25;
opts.numSubBatches  = 5;
 
opts.imdb           = imdb;
opts.smp            = 1;
opts.layerFile = str2func(['cnn_sparse_view_init_multi']);

gpus                = 2;

expDir          = ['data/'];
expDir          = [expDir func2str(opts.layerFile) '_' opts.method];
addDir          = ['_input' num2str(opts.inputSize(1))];

expDir          = [expDir, addDir];

opts.expDir = expDir;

opts.train   	= struct('gpus', gpus);

%%
[net_train, info_train]         = cnn_sparse_view_train(opts);
return ;
