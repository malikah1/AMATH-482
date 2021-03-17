%% Set up
clear; close all; clc;

%% Import Floyd Song
figure(1)
[yf, Fs] = audioread('Floyd.m4a');
tr_pf = length (yf) / Fs; % record time in seconds
plot((1:length(yf)) / Fs, yf);
xlabel('Time [sec]');
ylabel('Amplitude');
title('Comfortably Numb');
% p8 = audioplayer(yf, Fs); playblocking(p8);

%% Import GNR Song 
figure(2)
[yG, Gs] = audioread('GNR.m4a');
tr_gna = length (yG) / Gs; % record time in seconds
plot((1:length(yG)) / Gs, yG);
xlabel('Time [sec]');
ylabel('Amplitude');
title('Sweet Child O'' Mine');
p9 = audioplayer(yG, Gs); playblocking(p9);

%% Reproduce Music Score for GNR Song
L = tr_gna;
n = length(yG);
t2 = linspace(0,L,n+1);
t = t2(1:n);

k = (1/L)*[0:n/2-1 -n/2:-1];
ks = fftshift(k);

a = 100;
tau = 0:0.1:L;

for j = 1:length(tau)
    g = exp(-a*(t-tau(j)).^2); % Window function
    Sg = g'.* yG;
    Sgt = fft(Sg);
    [M, I] = max(abs(Sgt));
    notes(I) = abs(ks(I));
%     Sgt_spec(:,j) = fftshift(abs(Sgt));
end

% figure(3)
% pcolor(tau, ks, log(abs(Sgt_spec+1))) 
% shading interp
% set(gca, 'ylim', [0, 1000],'Fontsize', 16)
% colormap(hot)
% colorbar
% xlabel('time (t)'), ylabel('frequency (Hz)')

%% Reproduce Music Score for PF Song
L = tr_pf;
n = length(yf);
t2 = linspace(0,L,n+1);
t = t2(1:n);

k = (1/L)*[0:n/2-1 -n/2:-1];
ks = fftshift(k);

a = 100;
tau = 0:0.5:L;

for j = 1:length(tau)
    g = exp(-a*(t-tau(j)).^2); % Window function
    Sg = g'.* yf;
    Sgt = fft(Sg);
    Sgt_spec(:,j) = fftshift(abs(Sgt));
end

Sgt_spec = Sgt_spec(1:length(ks), :);

figure(4)
pcolor(tau, ks, log(abs(Sgt_spec + 1)))
shading interp
set(gca, 'ylim', [0, 300],'Fontsize', 16)
colormap(hot)
colorbar
xlabel('time (t)'), ylabel('frequency (Hz)')

%% Part 2, taking out the overtones of the bass

L = tr_pf;
n = length(yf);
t2 = linspace(0,L,n+1);
t = t2(1:n);
k = (1/L)*[0:n/2-1 -n/2:-1];
ks = fftshift(k);

a = 100;
tau = 0:0.5:L;

ti = 0.2;

for j = 1:length(tau)
    g = exp(-a*(t-tau(j)).^2); % Window function
    Sg = g'.* yf;
    Sgt = fft(Sg);
    [M, I] = max(abs(Sgt));
    
    Sgt = Sgt(1:length(ks), :);
    Sgtft = Sgt'.* exp(-ti*(k-k(I)).^2);
    
    Sgt_spec(:,j) = fftshift(abs(Sgtft));
end

figure(5)
pcolor(tau, ks, log(abs(Sgt_spec+1)))
shading interp
set(gca, 'ylim', [0, 300],'Fontsize', 16)
colormap(hot)
colorbar
xlabel('time (t)'), ylabel('frequency (Hz)')

%% Part 3

%importing new track
[yf, Fs] = audioread('Floyd2.m4a');
tr_pf = length (yf) / Fs; % record time in seconds
plot((1:length(yf)) / Fs, yf);
xlabel('Time [sec]');
ylabel('Amplitude');
title('Comfortably Numb');
% p8 = audioplayer(yf, Fs); playblocking(p8);

%% Part 3 actually trying to do something
L = tr_pf;
n = length(yf);
t2 = linspace(0,L,n+1);
t = t2(1:n);
k = (1/L)*[0:n/2-1 -n/2:-1];
ks = fftshift(k);

a = 100;
tau = 0:0.1:L;

ti = 0.2;

for j = 1:length(tau)
    g = exp(-a*(t-tau(j)).^2); % Window function
    Sg = g'.* yf;
    Sgt = fft(Sg);
    [M, I] = max(abs(Sgt));
    
    Sgt = Sgt(1:length(ks), :);
    Sgtft = Sgt'.* exp(-ti*(k-k(I)).^2);
    
    Sgt_spec(:,j) = fftshift(abs(Sgtft));
end

figure(4)
pcolor(tau, ks, log(abs(Sgt_spec + 1)))
shading interp
set(gca, 'ylim', [300, 1000],'Fontsize', 16)
colormap(hot)
colorbar
xlabel('time (t)'), ylabel('frequency (Hz)')

