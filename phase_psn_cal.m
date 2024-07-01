clear;
%close all;
clc;
%% parameter
lamda=532.0e-6;
pix=0.00374;
k=2*pi/lamda;
oz=30;
od=-oz;
%% complex
F1=imread('Baboon.bmp');
%F1=rgb2gray(F1);
F1=imresize(F1,[1024,1024]);
F1=im2double(F1);
F1=F1/max(max(F1));
[n1,m1]= size(F1);
%% zero padding
% k11=1100;
% k22=1100;
% K11=0.5*ones(k22,(k11-m1)/2);
% K22=0.5*ones((k22-n1)/2,m1);
% F1=[K11,[K22;F1;K22],K11];
% [nn,mm]=size(F1);
% k1=3840;
% k2=2160;
% K1=zeros(k2,(k1-mm)/2);
% K2=zeros((k2-nn)/2,mm);
% F1=[K1,[K2;F1;K2],K1];
[nn,mm]=size(F1);
F1=padarray(F1,[nn/2,mm/2]);
[N,M]=size(F1);
C1=F1;
%% grid
M1=zeros(N,M);
M2=ones(N,M);
dn=1;
n=N/dn;
m=M/dn;
sma=zeros(n,m);
for x=1:n
    for y=1:m
        if mod(x+y,2)==1
            sma(x,y)=1;
        end
    end
end
for i=1:n
    for j=1:m
        M1(i,j)=sma(ceil(i/dn),ceil(j/dn));
    end
end
M2=M2-M1;
%%
load ps
load pn
load cir
d=1200;
ps=imresize(ps,[N,M]);
pn=imresize(ps,[N,M]);
cir=imresize(cir,[d,d]);
filter2=zeros(N,M);
filter2(((N/2)-(d/2)+1):((N/2)+(d/2)),((M/2)-(d/2)+1):((M/2)+(d/2)))=cir;
%figure,imshow(filter,[]);
% imwrite(ps,'boat-ps.bmp');
%% diffraction
[fx,fy]=meshgrid(linspace(-1/(2*pix),1/(2*pix),M),linspace(-1/(2*pix),1/(2*pix),N));
H_AS=exp(1i*k*oz.*sqrt(1-(lamda*fx).^2-(lamda*fy).^2)); 
h_AS=exp(1i*k*od.*sqrt(1-(lamda*fx).^2-(lamda*fy).^2)); 
modi=cos(pi.*pix.*fx).*cos(pi.*pix.*fy);
% modi=sqrt(1-modi.^2);
% figure,imshow(mat2gray(modi),[]);
C1=fftshift(fft2(fftshift(C1)));
C1_o=C1.*H_AS;
%% Band-limitation 
C2_o=C1_o.*ps;
%C2_o=C1_o; %without limiation
H=fftshift(ifft2(fftshift(C2_o)));
%H=H(((N/2)-(nn/2)+1):((N/2)+(nn/2)),((M/2)-(mm/2)+1):((M/2)+(mm/2)));
spe4=mat2gray(log(1+abs(C2_o)));
%save('obj-spe4','spe4');
%figure,imshow(mat2gray(log(1+abs(C2_o))),[]);
%% complex encoding
A=abs(H);
A=A/max(max(A)); 
fai=angle(H);
fai=mod(fai,2*pi); 
sita1=fai-acos(A);
sita2=fai+acos(A);
dph=M2.*sita1+M1.*sita2;
dph=mod(dph,2*pi);
dph=mat2gray(dph);
% save('house-pn','dph');
% save('house','F1');
% imwrite(dph,'boat-dph.bmp');
%% 4-f system
g_pha=exp(1i*2*pi*dph);
%g_pha=padarray(g_pha,[nn/2,mm/2]);
G=fftshift(fft2(fftshift(g_pha)));
G1=filter2.*G;
%dphspe1=mat2gray(log(1+abs(G)));
%save('dph-spe1','dphspe1');
%figure,imshow(mat2gray(log(1+abs(G))),[]);
g_com=fftshift(ifft2(fftshift(G1)));
%% back propogation
[fx,fy]=meshgrid(linspace(-1/(2*pix),1/(2*pix),M),linspace(-1/(2*pix),1/(2*pix),N));
f_spe=fftshift(fft2(fftshift(g_com)));
f_spe2=f_spe.*h_AS;
u=fftshift(ifft2(ifftshift(f_spe2)));
rec=abs(u);
rec=rec(((N/2)-(n1/2)+1):((N/2)+(n1/2)),((M/2)-(m1/2)+1):((M/2)+(m1/2)));
figure,imshow(rec,[]);
%%
F1=F1(((N/2)-(n1/2)+1):((N/2)+(n1/2)),((M/2)-(m1/2)+1):((M/2)+(m1/2)));
recp=rec(22:152,156:286);
recp=mat2gray(recp);
recp2=rec(835:970,835:970);
recp2=mat2gray(recp2);
recp3=rec(716:835,810:929);
recp3=mat2gray(recp3);
figure,imshow(recp,[]);
figure,imshow(recp2,[]);
figure,imshow(recp3,[]);
% imwrite(recp2,'recps-baboon-1350-part1.bmp');
% imwrite(recp3,'recps-baboon-1350-part2.bmp');
% imwrite(recp,'recps-baboon-1350-part.bmp');
F1=mat2gray(F1);
rec=mat2gray(rec);
err=abs(F1-rec);
% save('err-baboon-000','err');
%figure,imshow(F1,[]);
%figure,imshow(rec,[]);
Diff=255*double(F1)-255*double(rec);
MSE=sum(Diff(:).^2)/numel(F1);
RMSE2=sqrt(MSE);
PSNR=10*log10(255^2/MSE);
SSIM=ssim(rec,F1);
