The following code was used to classify images based on two types of noises viz:- diffused noise and white noise





clc
close all;
clear all;

sf=dir('Original\im*');
for i = 1 : length(sf)
fn=strcat(pwd,'\Original\',sf(i).name);
org{i}=imread(fn);
end

sf1 = dir('10NL\*aA*');
for i = 1 : length(sf)
fn1=strcat(pwd,'\10NL\',sf1(i).name);
loc{i}=imread(fn1);
end

sf2=dir('10NL\*qA*');
for i = 1 : length(sf2)
fn2=strcat(pwd,'\10NL\',sf2(i).name);
dif{i}=imread(fn2);
end
nImages=280;

for i=1:nImages
  x(i,1)= psnr(loc{i}, org{i});
  x(i,2)=std2(loc{i});%standard dev. of localized noise
  I=im2bw(loc{i});
  A=regionprops(I, 'Area');
  x(i,3)=size(A,1);
  x(i,4)=mean(stdfilt(loc{i}(:)));%intensity of localized noise
  
  x(i+nImages,1)= psnr(dif{i}, org{i});
  x(i+nImages,2)=std2(dif{i});%standard dev. of diffused noise
  I1=im2bw(dif{i});
  A1=regionprops(I1, 'Area');
  x(i+nImages,3)=size(A1,1);
 x(i+nImages,4)=mean(stdfilt(dif{i}(:)));%standard dev. of diffused noise
  
  y(i,1)=cellstr('Sparse');
  y(i+nImages,1)=cellstr('Diffused');
end

M1 = fitNaiveBayes(x,y)
% 
% 
predictLabels1 = predict(M1,x);
[ConfusionMat1,labels] = confusionmat(y,predictLabels1)
