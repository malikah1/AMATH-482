clear; close all; clc;

%%
[images, labels] = mnist_parse('train-images-idx3-ubyte', 'train-labels-idx1-ubyte');
[timages, tlabels] = mnist_parse('t10k-images-idx3-ubyte', 't10k-labels-idx1-ubyte');
%%
images = im2double(images);
[labels,I] = sort(labels);

A = zeros(784,60000);
for i = 1:60000
    A(:,i) = reshape(images(:,:,i), 1, []);
end

A = A(:,I);
[U,S,V] = svd(A, 'econ');
sig = diag(S);
energies = zeros(1, length(sig));
for j = 1:length(sig)
    energies(j) = sig(j)^2/sum(sig.^2);
end
%%
timages = im2double(timages);
[tlabels, I] = sort(tlabels);

X = zeros(784,10000);
for i = 1:10000
    X(:,i) = reshape(timages(:,:,i), 1, []);
end

X = X(:, I);
%%
for i = 1:length(sig)
    cum(i) = sum(energies(1:i));
end
figure(1)
x= U(:,1:53)*S(1:53,1:53)*V(:,1:53)';
imshow(reshape(x(:,1), 28, 28))
figure(2)
x2= U(:,1:103)*S(1:103,1:103)*V(:,1:103)';
imshow(reshape(x2(:,1), 28, 28))
%%
plot3(U(:,2)'*A(:, 1:5923), U(:,3)'*A(:, 1:5923), U(:,4)'*A(:, 1:5923),'go');
hold on;
dR = [0.5, 0, 0];
dG = [0, 0.5, 0];
dB = [0, 0, 0.5];
plot3(U(:,2)'*A(:, 5924:12665), U(:,3)'*A(:, 5924:12665), U(:,4)'*A(:, 5924:12665),'mo');
plot3(U(:,2)'*A(:, 12666:18623), U(:,3)'*A(:, 12666:18623), U(:,4)'*A(:, 12666:18623),'co');
plot3(U(:,2)'*A(:, 18624:24754), U(:,3)'*A(:, 18624:24754), U(:,4)'*A(:, 18624:24754),'ro');
plot3(U(:,2)'*A(:, 24755:30596), U(:,3)'*A(:, 24755:30596), U(:,4)'*A(:, 24755:30596),'yo');
plot3(U(:,2)'*A(:, 30597:36017), U(:,3)'*A(:, 30597:36017), U(:,4)'*A(:, 30597:36017),'bo');
plot3(U(:,2)'*A(:, 36018:41935), U(:,3)'*A(:, 36018:41935), U(:,4)'*A(:, 36018:41935),'o', 'Color', dB);
plot3(U(:,2)'*A(:, 41936:48200), U(:,3)'*A(:, 41936:48200), U(:,4)'*A(:, 41936:48200),'ko');
plot3(U(:,2)'*A(:, 48201:54051), U(:,3)'*A(:, 48201:54051), U(:,4)'*A(:, 48201:54051),'o', 'Color', dR);
plot3(U(:,2)'*A(:, 54052:60000), U(:,3)'*A(:, 54052:60000), U(:,4)'*A(:, 54052:60000),'o', 'Color', dG);
xlabel('Projection onto 2nd Singular Vector');
ylabel('Projection onto 3rd Singular Vector');
zlabel('Projection onto 4th Singular Vector');
legend('digit 0', 'digit 1','digit 2', 'digit 3','digit 4','digit 5','digit 6','digit 7','digit 8','digit 9');
%% 2 Digit LDA
feature = 103;

o = A(:, 5924:12665);
e = A(:, 48201:54051);

n1 = size(o,2);
n8 = size(e, 2);

nums = U'*A;
os = nums(1:feature, 5924:12665);
es = nums(1:feature, 48201:54051);

m1 = mean(os,2);
m8 = mean(es, 2);

Sw = 0;
for k = 1:n1
    Sw = Sw + (os(:,k) - m1)*(os(:,k) - m1)';
end

for k = 1:n8
   Sw = Sw + (es(:,k) - m8)*(es(:,k) - m8)';
end

Sb = (m1-m8)*(m1-m8)';

[V2, D] = eig(Sb, Sw);
[lambda, ind] = max(abs(diag(D)));
w = V2(:, ind);
w = w/norm(w,2);

vone = w'*os;
veight = w'*es;

if mean(vone) > mean(veight)
    w = -w;
    vone = -vone;
    veight = -veight;
end
%%
plot(vone, zeros(n1), 'ob', 'Linewidth', 2)
hold on
plot(veight, ones(n8), 'dr', 'Linewidth', 2)
%%
sortones = sort(vone);
sorteights = sort(veight);

t1 = length(sortones);
t2 = 1;
while sortones(t1) > sorteights(t2)
    t1 = t1 - 1;
    t2 = t2 + 1;
end

threshold = (sortones(t1) + sorteights(t2))/2;
%%
subplot(1,2,1)
histogram(sortones,30); 
hold on, plot([threshold threshold], [0 1500],'r')
set(gca,'Xlim',[-3 4],'Ylim',[0 1500],'Fontsize',14)
title('ones')
subplot(1,2,2)
histogram(sorteights,30); hold on, plot([threshold threshold], [0 600],'r')
set(gca,'Xlim',[-3 4],'Ylim',[0 600],'Fontsize',14)
title('eights')
%% 2 DIGIT TEST DATA
feature = 103;
training = zeros(12593,feature);
ma = U'*A;
training(1:6742, :) = ma(1:feature, 5924:12665)';
training(6743:12593, :) = ma(1:feature, 48201:54051)';

group = zeros(12593, 1);
group(1:6742, 1) = labels(5924:12665,1);
group(6743:12593, 1) = labels(48201:54051,1);

nums = U'*X;
sample = zeros(2109, feature);
sample(1:1135,:) = nums(1:feature, 981:2115)';
sample(1136:2109,:) = nums(1:feature, 8018:8991)';

[class2,err2] = classify(sample, training, group, 'linear');

checks = zeros(2109,1);
checks(1:1135, 1) = 1;
checks(1136:2109, 1) = 8;

error = 0;
for i = 1:2109
    if class2(i) ~= checks(i)
        error = error + 1;
    end
end
error = error/2109;
%% 3 DIGIT LDA
feature = 103;
training = zeros(18014,feature);
ma = U'*A;
training(1:6742, :) = ma(1:feature, 5924:12665)';
training(6743:12593, :) = ma(1:feature, 48201:54051)';
training(12594:18014, :) = ma(1:feature, 30597:36017)';

group = zeros(18014, 1);
group(1:6742, 1) = labels(5924:12665,1);
group(6743:12593, 1) = labels(48201:54051,1);
group(12594:18014, 1) = labels(30597:36017,1);

nums = U'*X;
sample = zeros(3001, feature);
sample(1:1135,:) = nums(1:feature, 981:2115)';
sample(1136:2109,:) = nums(1:feature, 8018:8991)';
sample(2110:3001,:) = nums(1:feature, 5140:6031)';

[class1,err1] = classify(training, training, group, 'linear');
[class2, err2] = classify(sample, training, group, 'linear');

checks = zeros(3001,1);
checks(1:1135, 1) = 1;
checks(1136:2109, 1) = 8;
checks(2110:3001, 1) = 5;
error = 0;
for i = 1:3001
    if class2(i) ~= checks(i)
        error = error + 1;
    end
end
error = error/3001;
%% 2 DIGIT LDA - Bullet 3 (7/9)
feature = 103;
training = zeros(12214,feature);
ma = U'*A;
training(1:6265,:) = ma(1:feature, 41936:48200)';
training(6266:12214,:) = ma(1:feature, 54052:60000)';

group = zeros(12214, 1);
group(1:6265, 1) = labels(41936:48200,1);
group(6266:12214, 1) = labels(54052:60000,1);

[class1, err1] = classify(training, training, group, 'linear');

sample = zeros(2037, feature);
nums = U'*X;
sample(1:1028,:) = nums(1:feature, 6990:8017)';
sample(1029:2037,:) = nums(1:feature, 8992:10000)';

[class2, err2] = classify(sample, training, group, 'linear');
%%
feature = 103;
training = zeros(12214,feature);
ma = U'*A;
training(1:6265,:) = ma(1:feature, 24755:30596)';
training(6266:12214,:) = ma(1:feature, 54052:60000)';

group = zeros(12214, 1);
group(1:6265, 1) = labels(24755:30596,1);
group(6266:12214, 1) = labels(54052:60000,1);

[class1, err1] = classify(training, training, group, 'linear');

sample = zeros(2037, feature);
nums = U'*X;
sample(1:1028,:) = nums(1:feature, 4158:5139)';
sample(1029:2037,:) = nums(1:feature, 8992:10000)';

[class2, err2] = classify(sample, training, group, 'linear');
%% 2 DIGIT LDA - Bullet 4 (0/1)
feature = 103;

training = zeros(12665,feature);
ma = U'*A;
training(1:5923,:) = ma(1:feature, 1:5923)';
training(5924:12665,:) = ma(1:feature, 5924:12665)';

group = zeros(12665, 1);
group(1:5923, 1) = labels(1:5923,1);
group(5924:12665, 1) = labels(5924:12665,1);

[class1, err1] = classify(training, training, group, 'linear');

sample = zeros(2115, feature);
nums = U'*X;
sample(1:980,:) = nums(1:feature, 1:980)';
sample(981:2115,:) = nums(1:feature, 981:2115)';

[class2, err2] = classify(sample, training, group, 'linear');
%% BULLET 5 UGHHHHHHHHHH 
feature = 103;
ma = U'*A;
training = ma(1:feature, :)';
nums = U'*X;
test = nums(1:feature, :)';

tree = fitctree(training, labels,'MaxNumSplits', 103, 'CrossVal', 'on');
view(tree.Trained{1}, 'Mode', 'graph');
classError = kfoldLoss(tree);
%%
mdl = fitcecoc(training, labels);
test_labels = predict(mdl,test);
error = 0;
for i = 1:10000
    if test_labels(i) ~= tlabels(i)
        error = error + 1;
    end
end
error = error/10000;
%% BULLET 6 !!!!!1
% hardest two digits
feature = 103;
training = zeros(12214,feature);
ma = U'*A;
training(1:6265,:) = ma(1:feature, 41936:48200)';
training(6266:12214,:) = ma(1:feature, 54052:60000)';

group = zeros(12214, 1);
group(1:6265, 1) = labels(41936:48200,1);
group(6266:12214, 1) = labels(54052:60000,1);

tree = fitctree(training, group, 'MaxNumSplits', 103, 'CrossVal', 'on');
view(tree.Trained{1}, 'Mode', 'graph');
classError = kfoldLoss(tree);

sample = zeros(2037, feature);
nums = U'*X;
sample(1:1028,:) = nums(1:feature, 6990:8017)';
sample(1029:2037,:) = nums(1:feature, 8992:10000)';

checks = zeros(2037,1);
checks(1:1028, 1) = 7;
checks(1029:2037, 1) = 9;

mdl = fitcecoc(training, group);
test_labels = predict(mdl,sample);
error = 0;
for i = 1:2037
    if test_labels(i) ~= checks(i)
        error = error + 1;
    end
end
error = error/2037;
%% Easiest two (Bullet 6)
feature = 103;

training = zeros(12665,feature);
ma = U'*A;
training(1:5923,:) = ma(1:feature, 1:5923)';
training(5924:12665,:) = ma(1:feature, 5924:12665)';

group = zeros(12665, 1);
group(1:5923, 1) = labels(1:5923,1);
group(5924:12665, 1) = labels(5924:12665,1);

tree = fitctree(training, group, 'MaxNumSplits', 103, 'CrossVal', 'on');
view(tree.Trained{1}, 'Mode', 'graph');
classError = kfoldLoss(tree);

sample = zeros(2115, feature);
nums = U'*X;
sample(1:980,:) = nums(1:feature, 1:980)';
sample(981:2115,:) = nums(1:feature, 981:2115)';

checks = zeros(2115,1);
checks(1:980, 1) = 0;
checks(981:2115, 1) = 1;

mdl = fitcecoc(training, group);
test_labels = predict(mdl,sample);
error = 0;
for i = 1:2115
    if test_labels(i) ~= checks(i)
        error = error + 1;
    end
end
error = error/2115;
%%
function [images, labels] = mnist_parse(path_to_digits, path_to_labels)

% The function is curtesy of stackoverflow user rayryeng from Sept. 20,
% 2016. Link: https://stackoverflow.com/questions/39580926/how-do-i-load-in-the-mnist-digits-and-label-data-in-matlab

% Open files
fid1 = fopen(path_to_digits, 'r');

% The labels file
fid2 = fopen(path_to_labels, 'r');

% Read in magic numbers for both files
A = fread(fid1, 1, 'uint32');
magicNumber1 = swapbytes(uint32(A)); % Should be 2051
fprintf('Magic Number - Images: %d\n', magicNumber1);

A = fread(fid2, 1, 'uint32');
magicNumber2 = swapbytes(uint32(A)); % Should be 2049
fprintf('Magic Number - Labels: %d\n', magicNumber2);

% Read in total number of images
% Ensure that this number matches with the labels file
A = fread(fid1, 1, 'uint32');
totalImages = swapbytes(uint32(A));
A = fread(fid2, 1, 'uint32');
if totalImages ~= swapbytes(uint32(A))
    error('Total number of images read from images and labels files are not the same');
end
fprintf('Total number of images: %d\n', totalImages);

% Read in number of rows
A = fread(fid1, 1, 'uint32');
numRows = swapbytes(uint32(A));

% Read in number of columns
A = fread(fid1, 1, 'uint32');
numCols = swapbytes(uint32(A));

fprintf('Dimensions of each digit: %d x %d\n', numRows, numCols);

% For each image, store into an individual slice
images = zeros(numRows, numCols, totalImages, 'uint8');
for k = 1 : totalImages
    % Read in numRows*numCols pixels at a time
    A = fread(fid1, numRows*numCols, 'uint8');

    % Reshape so that it becomes a matrix
    % We are actually reading this in column major format
    % so we need to transpose this at the end
    images(:,:,k) = reshape(uint8(A), numCols, numRows).';
end

% Read in the labels
labels = fread(fid2, totalImages, 'uint8');

% Close the files
fclose(fid1);
fclose(fid2);

end
