function cnn_display(res, net, ifig)

%%
inputRange      = net.meta.trainOpts.inputRange;
offset          = net.meta.trainOpts.offset;
wgt             = net.meta.trainOpts.wgt;

lv              = net.meta.trainOpts.lv;
pflt            = net.meta.trainOpts.pflt;
dflt            = net.meta.trainOpts.dflt;
% nch           	= sum([1; 2.^lv(:)]);

% dc              = net.layers{end}.dc(:,:,:,1);
dc              = 0;

if (strcmp(net.meta.trainOpts.method, 'normal'))
    labels          = gather(net.layers{end}.class(:,:,:,1));
    recon           = gather(res(end - 1).x(:,:,:,1));
    data            = gather(res(1).x(:,:,:,1));
    
%     err_labels      = data - labels;
    err_recon       = recon - labels;
    
%     err_labels      = ((err_labels(:,:,:) + dc) - offset)./wgt;
    err_recon       = ((err_recon(:,:,:) + dc) - offset)./wgt;
    
    labels(:,:,:)	= ((labels(:,:,:) + dc) - offset)./wgt;
    data(:,:,:)     = ((data(:,:,:) + dc) - offset)./wgt;
    recon(:,:,:)	= ((recon(:,:,:) + dc) - offset)./wgt;
else
    err_labels      = gather(net.layers{end}.class(:,:,:,1));
    err_recon       = gather(res(end - 1).x(:,:,:,1));
    
    data            = gather(res(1).x(:,:,:,1));
    labels          = data - err_labels;
    recon           = data - err_recon;
    
    err_labels      = ((err_labels(:,:,:) + dc) - offset)./wgt;
    err_recon       = ((err_recon(:,:,:) + dc) - offset)./wgt;
    
    labels(:,:,:)	= ((labels(:,:,:) + dc) - offset)./wgt;
    data(:,:,:)     = ((data(:,:,:) + dc) - offset)./wgt;
    recon(:,:,:)	= ((recon(:,:,:) + dc) - offset)./wgt;
end


if ~isempty(lv)
    
%     pflt = 'pyr';                % Pyramidal filter
%     dflt = 'vk';                 %'cd' ;  % Directional filter
    
%     for iz = 1:nz
    for iz = 1
        
        % Nonsubsampled Contourlet decomposition
        data        = nsctrec(mat2wavecell( double(data(:,:,:,iz)), lv ), dflt, pflt);
        labels      = nsctrec(mat2wavecell( double(labels(:,:,:,iz)), lv ), dflt, pflt);
        recon       = nsctrec(mat2wavecell( double(recon(:,:,:,iz)), lv ), dflt, pflt);
        err_labels	= nsctrec(mat2wavecell( double(err_labels(:,:,:,iz)), lv ), dflt, pflt);
        err_recon	= nsctrec(mat2wavecell( double(err_recon(:,:,:,iz)), lv ), dflt, pflt);

    end
    
end

maxVal              = 1/max(labels(:));

data_               = data*maxVal;

labels_             = labels*maxVal;
recon_              = recon*maxVal;

labels_(labels_ < 0)	= 0;
data_(data_ < 0)        = 0;
recon_(recon_ < 0)      = 0;

wnd                     = inputRange;

%%
% psnr_data	= psnr(double(data_),       double(labels_));
psnr_cnn1    = psnr(double(recon_(:,:,1)),      double(labels_(:,:,1)));
psnr_cnn2    = psnr(double(recon_(:,:,2)),      double(labels_(:,:,2)));

% ssim_data   = ssim(double(data_),       double(labels_));
ssim_cnn1    = ssim(double(recon_(:,:,1)),      double(labels_(:,:,1)));
ssim_cnn2    = ssim(double(recon_(:,:,2)),      double(labels_(:,:,2)));

% mse_data    = immse(double(data_),      double(labels_));
mse_cnn1     = immse(double(recon_(:,:,1)),     double(labels_(:,:,1)));
mse_cnn2     = immse(double(recon_(:,:,2)),     double(labels_(:,:,2)));

% figure(ifig);    colormap(gray(256));
% subplot(231);   imagesc(labels(:,:,1), wnd);       axis image off;     title('LABELS');
% subplot(232);   imagesc(data(:,:,1), wnd);         axis image off;     title({['DATA'];	['PSNR = '   num2str(psnr_data, '%.6f') 'dB'];	['MSE = ' num2str(mse_data, '%.6f')];	['SSIM = ' num2str(ssim_data, '%.6f')]});
% subplot(233);   imagesc(recon(:,:,1), wnd);        axis image off;     title({['RECON']; 	['PSNR = '	num2str(psnr_cnn,    '%.6f') 'dB'];	['MSE = ' num2str(mse_cnn, '%.6f')];	['SSIM = ' num2str(ssim_cnn,  '%.6f')]});
% subplot(234);   imagesc(recon(:,:,1) - labels(:,:,1));	axis image off;     title({['ERR_{DIFF}']});
% subplot(235);   imagesc(err_labels(:,:,1));        axis image off;     title({['ERR_{DATA}']});
% subplot(236);   imagesc(err_recon(:,:,1));         axis image off;     title({['ERR_{RECON}']});

figure(ifig);    colormap(gray(256));
subplot(221);   imagesc(labels(:,:,1), wnd);       axis image off;     title('LABELS Real');
subplot(222);   imagesc(labels(:,:,2), wnd);       axis image off;     title('LABELS Imag');
% subplot(131);   imagesc(data(:,:,2), wnd);        axis image off;     title({['DATA']; 	['PSNR = '	num2str(psnr_data,    '%.6f') 'dB'];	['MSE = ' num2str(mse_data, '%.6f')];	['SSIM = ' num2str(ssim_data,  '%.6f')]});
% subplot(223);   imagesc(data(:,:,32), wnd);         axis image off;     title({['DATA CH2']});
subplot(223);   imagesc(recon(:,:,1), wnd);        axis image off;     title({['RECON Real']; 	['PSNR = '	num2str(psnr_cnn1,    '%.6f') 'dB'];	['MSE = ' num2str(mse_cnn1, '%.6f')];	['SSIM = ' num2str(ssim_cnn1,  '%.6f')]});
subplot(224);   imagesc(recon(:,:,2), wnd);        axis image off;     title({['RECON Imag']; 	['PSNR = '	num2str(psnr_cnn2,    '%.6f') 'dB'];	['MSE = ' num2str(mse_cnn2, '%.6f')];	['SSIM = ' num2str(ssim_cnn2,  '%.6f')]});
% subplot(234);   imagesc(recon(:,:,1) - labels(:,:,1));	axis image off;     title({['ERR_{DIFF}']});
% subplot(235);   imagesc(err_labels(:,:,1));        axis image off;     title({['ERR_{DATA}']});
% subplot(236);   imagesc(err_recon(:,:,1));         axis image off;     title({['ERR_{RECON}']});





