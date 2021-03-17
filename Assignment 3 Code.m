clear; close all; clc

%%
load('cam1_1.mat');
load('cam2_1.mat');
load('cam3_1.mat');
%%
numFrames = size(vidFrames1_1,4);
x1 = zeros(1, numFrames);
y1 = zeros(1, numFrames);
fx = 220;
fy = 300;
for j = 1:numFrames
    X = vidFrames1_1(:,:,:,j);
    Xg = rgb2gray(X);
    Xg = im2double(Xg);
    filter = zeros(480,640);
    filter(((fx-30):(fx+30)), ((fy-30):(fy+30))) = 1;
    Xgf = Xg.*filter;
    [M,I] = max(Xgf(:));
    [x1(j),y1(j)] = ind2sub(size(Xgf), I);
    fx = x1(j);
    fy = y1(j);
    %imshow(Xgf); drawnow;
end

numFrames = size(vidFrames2_1,4);
x2 = zeros(1, numFrames);
y2 = zeros(1, numFrames);
fx = 276;
fy = 278;
for j = 1:numFrames
    X = vidFrames2_1(:,:,:,j);
    Xg = rgb2gray(X);
    Xg = im2double(Xg);
    filter = zeros(480,640);
    filter(((fx-30):(fx+30)), ((fy-30):(fy+30))) = 1;
    Xgf = Xg.*filter;
    [M,I] = max(Xgf(:));
    [x2(j),y2(j)] = ind2sub(size(Xgf), I);
    fx = x2(j);
    fy = y2(j);
    %imshow(Xgf); drawnow;
end

numFrames = size(vidFrames3_1,4);
x3 = zeros(1, numFrames);
y3 = zeros(1, numFrames);
fx = 273;
fy = 329;
for j = 1:numFrames
    X = vidFrames3_1(:,:,:,j);
    Xg = rgb2gray(X);
    Xg = im2double(Xg);
    filter = zeros(480,640);
    filter(((fx-30):(fx+30)), ((fy-30):(fy+30))) = 1;
    Xgf = Xg.*filter;
    [M,I] = max(Xgf(:));
    [x3(j),y3(j)] = ind2sub(size(Xgf), I);
    fx = x3(j);
    fy = y3(j);
    %imshow(Xgf); drawnow;
end
%%
x1 = x1(:, 11:226);
y1 = y1(:, 11:226);
x2 = x2(:, 19:234);
y2 = y2(:, 19:234);
x3 = x3(:, 8:223);
y3 = y3(:, 8:223);

A = [x1 - mean(x1); y1 - mean(y1); x2 - mean(x2); y2 - mean(y2); x3 - mean(x3); y3 - mean(y3)];

[U,S,V] = svd(A, 'econ');
sig = diag(S);
energies = zeros(1, length(sig));
for j = 1:length(sig)
    energies(j) = sig(j)^2/sum(sig.^2);
end
%%
figure(1)
plot(energies, 'mo', 'LineWidth', 4, 'MarkerSize', 8)
xlabel('Singular Value')
ylabel('Energy (%)')
title('Energy per Singular Value')
print('T1-1.png', '-dpng');

figure(2)
plot(sig(1:2)'.*V(:,1:2),'LineWidth', 1)
legend('Component 1', 'Component 2')
xlabel('Time (Frames)')
ylabel('Amplitude')
title('Motions of Paint Can through Time')
print('T1-2.png', '-dpng');
%%
load('cam1_2.mat');
load('cam2_2.mat');
load('cam3_2.mat');
%%
numFrames = size(vidFrames1_2,4);
x1 = zeros(1, numFrames);
y1 = zeros(1, numFrames);
fx = 328;
fy = 285;
for j = 1:numFrames
    X = vidFrames1_2(:,:,:,j);
    Xg = rgb2gray(X);
    Xg = im2double(Xg);
    filter = zeros(480,640);
    filter(((fx-30):(fx+30)), ((fy-30):(fy+30))) = 1;
    Xgf = Xg.*filter;
    [M,I] = max(Xgf(:));
    [x1(j),y1(j)] = ind2sub(size(Xgf), I);
    fx = x1(j);
    fy = y1(j);
    %imshow(Xgf); drawnow; 
end

numFrames = size(vidFrames2_2,4);
x2 = zeros(1, numFrames);
y2 = zeros(1, numFrames);
fx = 316;
fy = 356;
for j = 1:numFrames
    X = vidFrames2_2(:,:,:,j);
    Xg = rgb2gray(X);
    Xg = im2double(Xg);
    filter = zeros(480,640);
    filter(((fx-30):(fx+30)), ((fy-30):(fy+30))) = 1;
    Xgf = Xg.*filter;
    [M,I] = max(Xgf(:));
    [x2(j),y2(j)] = ind2sub(size(Xgf), I);
    fx = x2(j);
    fy = y2(j);
    %imshow(Xgf); drawnow;
end

numFrames = size(vidFrames3_2,4);
x3 = zeros(1, numFrames);
y3 = zeros(1, numFrames);
fx = 244;
fy = 350;
for j = 1:numFrames
    X = vidFrames3_2(:,:,:,j);
    Xg = rgb2gray(X);
    Xg = im2double(Xg);
    filter = zeros(480,640);
    filter(((fx-30):(fx+30)), ((fy-30):(fy+30))) = 1;
    Xgf = Xg.*filter;
    [M,I] = max(Xgf(:));
    [x3(j),y3(j)] = ind2sub(size(Xgf), I);
    fx = x3(j);
    fy = y3(j);
    %imshow(Xgf); drawnow;
end
%%
x2 = x2(:, 1:314);
y2 = y2(:, 1:314);
x3 = x3(:, 1:314);
y3 = y3(:, 1:314);

A = [x1 - mean(x1); y1 - mean(y1); x2 - mean(x2); y2 - mean(y2); x3 - mean(x3); y3 - mean(y3)];

[U,S,V] = svd(A, 'econ');
sig = diag(S);
energies = zeros(1, length(sig));
for j = 1:length(sig)
    energies(j) = sig(j)^2/sum(sig.^2);
end

figure(1)
plot(energies, 'mo', 'LineWidth', 4, 'MarkerSize', 8)
xlabel('Singular Value')
ylabel('Energy (%)')
title('Energy per Singular Value')
print('T2-1.png', '-dpng');

figure(2)
plot(sig(1:3)'.*V(:,1:3),'LineWidth', 1)
legend('Component 1', 'Component 2', 'Component 3')
xlabel('Time (Frames)')
ylabel('Amplitude')
title('Motions of Paint Can through Time')
print('T2-2.png', '-dpng');

%%
load('cam1_3.mat');
load('cam2_3.mat');
load('cam3_3.mat');
%%
numFrames = size(vidFrames1_3,4);
x1 = zeros(1, numFrames);
y1 = zeros(1, numFrames);
fx = 285;
fy = 290;
for j = 1:numFrames
    X = vidFrames1_3(:,:,:,j);
    Xg = rgb2gray(X);
    Xg = im2double(Xg);
    filter = zeros(480,640);
    filter(((fx-30):(fx+30)), ((fy-30):(fy+30))) = 1;
    Xgf = Xg.*filter;
    [M,I] = max(Xgf(:));
    [x1(j),y1(j)] = ind2sub(size(Xgf), I);
    fx = x1(j);
    fy = y1(j);
    %imshow(X); drawnow; 
end

numFrames = size(vidFrames2_3,4);
x2 = zeros(1, numFrames);
y2 = zeros(1, numFrames);
fx = 238;
fy = 292;
for j = 1:numFrames
    X = vidFrames2_3(:,:,:,j);
    Xg = rgb2gray(X);
    Xg = im2double(Xg);
    filter = zeros(480,640);
    filter(((fx-30):(fx+30)), ((fy-30):(fy+30))) = 1;
    Xgf = Xg.*filter;
    [M,I] = max(Xgf(:));
    [x2(j),y2(j)] = ind2sub(size(Xgf), I);
    fx = x2(j);
    fy = y2(j);
    %imshow(Xgf); drawnow;
end

numFrames = size(vidFrames3_3,4);
x3 = zeros(1, numFrames);
y3 = zeros(1, numFrames);
fx = 228;
fy = 354;
for j = 1:numFrames
    X = vidFrames3_3(:,:,:,j);
    Xg = rgb2gray(X);
    Xg = im2double(Xg);
    filter = zeros(480,640);
    filter(((fx-30):(fx+30)), ((fy-30):(fy+30))) = 1;
    Xgf = Xg.*filter;
    [M,I] = max(Xgf(:));
    [x3(j),y3(j)] = ind2sub(size(Xgf), I);
    fx = x3(j);
    fy = y3(j);
   % imshow(Xgf); drawnow;
end
%%
x1 = x1(:, 1:237);
y1 = y1(:, 1:237);
x2 = x2(:, 1:237);
y2 = y2(:, 1:237);

A = [x1 - mean(x1); y1 - mean(y1); x2 - mean(x2); y2 - mean(y2); x3 - mean(x3); y3 - mean(y3)];

[U,S,V] = svd(A, 'econ');
sig = diag(S);
energies = zeros(1, length(sig));
for j = 1:length(sig)
    energies(j) = sig(j)^2/sum(sig.^2);
end

figure(1)
plot(energies, 'mo', 'LineWidth', 4, 'MarkerSize', 8)
xlabel('Singular Value')
ylabel('Energy (%)')
title('Energy per Singular Value')
print('T3-1.png', '-dpng');

figure(2)
plot(sig(1:4)'.*V(:,1:4),'LineWidth', 1)
legend('Component 1', 'Component 2', 'Component 3', 'Component 4')
xlabel('Time (Frames)')
ylabel('Amplitude')
title('Motions of Paint Can through Time')
print('T3-2.png', '-dpng');
%%
load('cam1_4.mat');
load('cam2_4.mat');
load('cam3_4.mat');
%%
numFrames = size(vidFrames1_4,4);
x1 = zeros(1, numFrames);
y1 = zeros(1, numFrames);
fx = 272;
fy = 382;
for j = 1:numFrames
    X = vidFrames1_4(:,:,:,j);
    Xg = rgb2gray(X);
    Xg = im2double(Xg);
    filter = zeros(480,640);
    filter(((fx-30):(fx+30)), ((fy-30):(fy+30))) = 1;
    Xgf = Xg.*filter;
    [M,I] = max(Xgf(:));
    [x1(j),y1(j)] = ind2sub(size(Xgf), I);
    fx = x1(j);
    fy = y1(j);
    %imshow(Xgf); drawnow; 
end

numFrames = size(vidFrames2_4,4);
x2 = zeros(1, numFrames);
y2 = zeros(1, numFrames);
fx = 252;
fy = 230;
for j = 1:numFrames
    X = vidFrames2_4(:,:,:,j);
    Xg = rgb2gray(X);
    Xg = im2double(Xg);
    filter = zeros(480,640);
    filter(((fx-30):(fx+30)), ((fy-30):(fy+30))) = 1;
    Xgf = Xg.*filter;
    [M,I] = max(Xgf(:));
    [x2(j),y2(j)] = ind2sub(size(Xgf), I);
    fx = x2(j);
    fy = y2(j);
    %imshow(Xgf); drawnow;
end

numFrames = size(vidFrames3_4,4);
x3 = zeros(1, numFrames);
y3 = zeros(1, numFrames);
fx = 214;
fy = 364;
for j = 1:numFrames
    X = vidFrames3_4(:,:,:,j);
    Xg = rgb2gray(X);
    Xg = im2double(Xg);
    filter = zeros(480,640);
    filter(((fx-30):(fx+30)), ((fy-30):(fy+30))) = 1;
    Xgf = Xg.*filter;
    [M,I] = max(Xgf(:));
    [x3(j),y3(j)] = ind2sub(size(Xgf), I);
    fx = x3(j);
    fy = y3(j);
    %imshow(Xgf); drawnow;
end
%%
x2 = x2(:, 1:392);
y2 = y2(:, 1:392);
x3 = x3(:, 1:392);
y3 = y3(:, 1:392);

A = [x1 - mean(x1); y1 - mean(y1); x2 - mean(x2); y2 - mean(y2); x3 - mean(x3); y3 - mean(y3)];

[U,S,V] = svd(A, 'econ');
sig = diag(S);
energies = zeros(1, length(sig));
for j = 1:length(sig)
    energies(j) = sig(j)^2/sum(sig.^2);
end

figure(1)
plot(energies, 'mo', 'LineWidth', 4, 'MarkerSize', 8)
xlabel('Singular Value')
ylabel('Energy (%)')
title('Energy per Singular Value')
print('T4-1.png', '-dpng');

figure(2)
plot(sig(1:4)'.*V(:,1:4),'LineWidth', 1)
legend('Component 1', 'Component 2', 'Component 3', 'Component 4')
xlabel('Time (Frames)')
ylabel('Amplitude')
title('Motions of Paint Can through Time')
print('T4-2.png', '-dpng');