clear; close all; clc;
%%
sd = VideoReader('ski_drop_low.mp4');
%%
sd_vid = read(sd);
[n m s s2] = size(sd_vid);
for j = 1: s2
    sd_reshape = double(reshape(sd_vid(:,:,1,j), n*m, 1));
    vid_sd(:,j) = sd_reshape;
end

X = vid_sd;
X1 = X(:,1:end-1);
X2 = X(:, 2:end);

[U,Sig,V] = svd(X1, 'econ');
S = U'*X2*V*diag(1./diag(Sig));

[eV, D] = eig(S);
mu = diag(D);
omega = log(mu);
Phi = U*eV;

t = 0.001;
b = find(abs(omega) < t);
omega_b = omega(b);
y0 = Phi(:,b)\X1(:,1);

u_modes = zeros(length(y0), 454);
for i = 1:454
   u_modes(:,i) = y0.*exp(omega_b);
end
u_dmd = Phi(:,b)*u_modes;

X_fg = X - abs(u_dmd);
R = zeros(length(X_fg), size(X_fg,2));
u_dmd = R + abs(u_dmd);
X_fg = X_fg - R;
X_fg = reshape(X_fg, [n m size(X,2)]);
u_dmd = reshape(u_dmd, [n m size(X,2)]);
%%
c_f = 200;
figure(1)
pcolor(flip(reshape(X(:, c_f), [n m]))); shading interp, colormap(gray)
figure(2)
pcolor(flip(u_dmd(:,:,c_f))); shading interp, colormap(gray)
figure(3)
pcolor(flip(X_fg(:,:,c_f))); shading interp, colormap(gray)