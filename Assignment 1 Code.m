%%
clear; close all; clc;

load subdata.mat

L = 10;
n = 64;

x2 = linspace(-L,L,n+1); 
x = x2(1:n); 
y = x; 
z = x;


k = (2*pi/(2*L))*[0:(n/2 - 1) -n/2:-1]; 
ks = fftshift(k);

[X,Y,Z] = meshgrid(x,y,z);
[Kx,Ky,Kz] = meshgrid(ks,ks,ks); % frequency of data in x,y,z direction

%% Not Needed
for j=1:49
    Un(:,:,:) = reshape(subdata(:,j),n,n,n);
    M = max(abs(Un),[],'all');
    close all, isosurface(X,Y,Z,abs(Un)/M,0.7)
    axis([-20 20 -20 20 -20 20]), grid on, drawnow
end

%% Averaging accross the spectrum to find center frequency
tot = 0;
for j = 1:49
    Un(:,:,:) = reshape(subdata(:,j),n,n,n);
    Unt = fftn(Un);
    tot = tot + Unt;
end

avg = abs(fftshift(tot))/49;

[argval, argmax] = max(abs(avg(:)));
[x1,y1,z1] = ind2sub(size(avg),argmax); % k values for center frequency, need to shift

%% Filter data around center frequency
tau = 0.1;
kx0 = Kx(x1,y1,z1);
ky0 = Ky(x1,y1,z1);
kz0 = Kz(x1,y1,z1);

filter = exp(-tau * ((Kx - kx0).^2 + (Ky - ky0).^2 + (Kz - kz0).^2));

 % apply filter in frequency space
 % convert back to time domain

% do this per realization, each time realization, then find max and then
% plot sense
%% Determine path for submarine + plot
xcords = zeros(1,49);
ycords = zeros(1,49);
for j = 1:49
    Un(:,:,:) = reshape(subdata(:,j),n,n,n); 
    
    Unft = filter.* fftshift(fftn((Un)));
    Unf = ifftn(Unft);
    
%     % Plot isosurface
%     M = max(abs(Unf(:)),[],'all');
%     isosurface(X,Y,Z,abs(Unf)/M,0.7)
%     axis([-20 20 -20 20 -20 20]), grid on, drawnow
    
    % Plot plot3
    [val, m] = max(abs(Unf(:)));
    [xf,yf,zf] = ind2sub(size(Unf),m);
    
    xfi = X(xf,yf,zf);
    yfi = Y(xf,yf,zf);
    zfi = Z(xf,yf,zf);
    
    xcords(j) = xfi;
    ycords(j) = yfi;
    
    plot3(xfi,yfi,zfi,'o','LineWidth', 5, 'MarkerSize', 10); hold on;
    axis([-20 20 -20 20 -20 20]), grid on, drawnow

    xlabel('X');
    ylabel('Y');
    zlabel('Z');
    title('Path of the Submarine', 'FontSize', 17);
end

% % Plot final point
% plot(xfi,yfi, 'o', 'LineWidth', 5, 'MarkerSize', 10, 'color', 'r');
% axis([-20 20 -20 20]), grid on, drawnow
% xlabel('X');
% ylabel('Y');
% title('Final Location of Submarine', 'FontSize', 17);





