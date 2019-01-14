function B_img=IQ2Bmode(rec);
rec=abs(rec')./max(abs(rec(:)));
N_ele = 192;
N_sc = 96;
pitch = 0.0200 * 1e-2;     % cm => m
scan_view_size = pitch*N_ele;
data_total = size(rec,1);
pixel_d = 1.9250e-05;
sc_d = scan_view_size/(N_sc); % Scanline distance

RF_env = abs(rec);

data_max = max(max(RF_env));

log_data = RF_env./data_max;

dB = 60;
min_dB = 10^(-dB/20);

for i=1:N_sc
    for j=1:data_total
        if(log_data(j,i) < min_dB)
            log_data(j,i) = 0;
        else
            log_data(j,i) = 255*((20/dB)*log10(log_data(j,i))+1);
        end
    end
end

%%
B_img = log_data;
% B_img = B_img(1:end*0.8,:);


