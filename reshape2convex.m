function [B_img]=reshape2convex(RFsum_LD);

   [data_total,N_sc]=size(RFsum_LD);
   dB = 60;
   min_dB = 10^(-dB/20);

   PI = 3.14;

info.c = 1540;
info.fs = 40e6;

info.N_ele = 192;%double(System.Transducer.elementCnt);
info.pitch = 0.000348;
info.ROC = 0.061;
info.pixel_d = info.c/info.fs/2;

info.N_sc = 96;
info.N_ch = 64;
AlignedSampleNum=size(RFsum_LD,1);
SampleNum=0.8*AlignedSampleNum;

Offset = 60;
data_total = double(AlignedSampleNum);
data_total1 = double(SampleNum);

info.RxFnum = 1;

info.RxFnum = 1;

info.FOV = 62.4300;
info.st_ang = -31.2150;
info.d_theta = info.FOV/(info.N_sc-1);

info.one_angle = info.FOV/(info.N_ele-1);

info.ScanAngle = [info.st_ang:info.d_theta:info.st_ang+(info.N_sc-1)*info.d_theta];
tmp = -info.FOV/2;
info.EleAngle = [tmp:info.one_angle:-tmp];
%% reordering information
rx_HalfCh = info.N_ch*0.5;
rx_ch_mtx = [-rx_HalfCh:rx_HalfCh-1];

RxMux = zeros(info.N_sc, info.N_ch);
SCvsEle = info.d_theta/info.one_angle;


 %%
% RF_env = abs(RF_Sum);
RF_log = RFsum_LD;%RF_env./max(max(RF_env));

idx = find(RF_log < min_dB);
RF_log = 255*((20/dB)*log10(RF_log)+1);
RF_log(idx) = 0;
%%
st_ang = info.st_ang*PI/180;
d_theta = info.d_theta*PI/180;

img_dep = data_total1*info.pixel_d + info.ROC*(1 - cos(st_ang));
img_width = 2 * (data_total*info.pixel_d + info.ROC) * sin(abs(st_ang));

img_z = 700;
img_x = round((img_width*img_z)/img_dep);

B_img = zeros(img_z, img_x);

spr_ang = (pi - abs(st_ang*2))/2;

dx = img_width/img_x;
dz = img_dep/img_z;

half_dist = -(img_x-1)/2*dx;


for i=1:img_x
    ix = half_dist + (i-1)*dx;
    for j=1:img_z
        iz = (j-1) * dz + info.ROC*cos(st_ang);
        
        Ang = atan2(ix,iz);
        
        z = (sqrt((ix^2)+(iz^2)) - info.ROC)/info.pixel_d;
        
        x = ((pi/2 - spr_ang) + Ang)/d_theta + 1;
        
        z_L = floor(z);
        z_H = z_L+1;
        x_L = floor(x);
        x_H = x_L+1;
        
        z_err = z-z_L;
        x_err = x-x_L;
        
        if((z_L>0) && (z_H <= data_total1) && (x_L > 0) &&(x_H <= info.N_sc))
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            Zon = RF_log(z_L,x_L);
            Zon1 = RF_log(z_H,x_L);
            Zin = RF_log(z_L,x_H);
            Zin1 = RF_log(z_H,x_H);
            
            Zri = Zin*(1-z_err) + Zin1*z_err;
            Zro = Zon*(1-z_err) + Zon1*z_err;
            Z = Zro*(1-x_err) + Zri*x_err;
            
            B_img(j,i) = Z;
        else
            B_img(j,i) = 255;
        end
    end
end