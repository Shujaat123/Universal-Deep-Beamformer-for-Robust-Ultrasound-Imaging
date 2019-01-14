function [net, info] = cnn_sparse_view_train(opts)

run(fullfile(fileparts(mfilename('fullpath')), 'matlab', 'vl_setupnn.m')) ;

if ~isfield(opts.train, 'gpus'), opts.train.gpus = []; end;

opts.imdbPath = fullfile(opts.dataDir, opts.dataFile);

% --------------------------------------------------------------------
%              Prepare data
% --------------------------------------------------------------------
net     = opts.layerFile( ...
    'batchNormalization',	opts.batchNormalization,	'networkType',      opts.networkType,	...
    'inputRange',       opts.inputRange,    ...
    'inputSize',            opts.inputSize,             'wrapSize',         opts.wrapSize,      ...
    'wgt',                  opts.wgt,                   'offset',           opts.offset,        ...
    'cwgt',                 opts.cwgt,                  'grdclp',           opts.grdclp,        ...
    'lrnrate',              opts.lrnrate,               'wgtdecay',         opts.wgtdecay,      ...
    'numEpochs',            opts.numEpochs,             'method',           opts.method,        ...
    'batchSize',            opts.batchSize,             'numSubBatches',	opts.numSubBatches, ...
    'lv',                   opts.lv,                    ...
    'pflt',                 opts.pflt,                  'dflt',             opts.dflt) ;

if ~exist(opts.expDir, 'dir')
    mkdir(opts.expDir);
end

if exist(opts.imdbPath, 'file')
    imdb	= opts.imdb;
else
    imdb = getMnistImdb(opts) ;
    mkdir(opts.expDir) ;
    save(opts.imdbPath, '-struct', 'imdb') ;
end

net.meta.classes.name = arrayfun(@(x)sprintf('%d',x),1:10,'UniformOutput',false) ;


% --------------------------------------------------------------------
%                       Train
% --------------------------------------------------------------------

switch opts.networkType
    case 'simplenn', trainfn  = @cnn_train ;
    case 'dagnn', trainfn     = @cnn_train_dag ;
end

[net, info] = trainfn(net, imdb, getBatch(opts), ...
    'expDir', opts.expDir, 'smp', opts.smp, ...
    net.meta.trainOpts, ...
    opts.train, ...
    'val', find(imdb.images.set == 3)) ;



% --------------------------------------------------------------------
function fn = getBatch(opts)
% --------------------------------------------------------------------
switch lower(opts.networkType)
    case 'simplenn'
        fn = @(x,y) getSimpleNNBatch(x,y,opts) ;
    case 'dagnn'
        bopts = struct('numGpus', numel(opts.train.gpus)) ;
        fn = @(x,y) getDagNNBatch(bopts,x,y) ;
end


% --------------------------------------------------------------------
function [data, labels, orig] = getSimpleNNBatch(imdb, batch, opts)
% --------------------------------------------------------------------
rng('shuffle');

patch   = opts.inputSize;
wgt     = opts.wgt;
offset  = opts.offset;
lv  	= opts.lv;
pflt    = opts.pflt;
dflt    = opts.dflt;

ny      = size(imdb.images.labels, 1);
nx      = size(imdb.images.labels, 2);
nch     = sum([1; 2.^lv(:)]);
nz      = length(batch);


iy      = 0;
ix      = floor(rand(1)*(nx - patch(2)));

by      = 1:patch(1);
bx      = 1:patch(2);

data_ 	= wgt*single(imdb.images.data(:,:,:,batch)) + offset ;
labels_	= wgt*single(imdb.images.labels(:,:,:,batch)) + offset ;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% CASE 1
data    = data_;
labels  = labels_;
orig    = labels_;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


if (imdb.images.set(1) == 1)
    if (rand > 0.5)
        data    = flip(data, 1);
        labels  = flip(labels, 1);
        orig    = flip(orig, 1);
    end
    
    if (rand > 0.5)
        data    = flip(data, 2);
        labels  = flip(labels, 2);
        orig    = flip(orig, 2);
    end
end

