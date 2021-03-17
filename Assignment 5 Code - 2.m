clear; close all; clc;
%%
mc = VideoReader('monte_carlo_low.mp4');

mc_vid = read(mc);
[n m s s2] = size(mc_vid);
for j = 1: s2
    mc_reshape = double(reshape(mc_vid(:,:,1,j), n*m, 1));
    vid_mc(:,j) = mc_reshape;
end

X = vid_mc;
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

u_modes = zeros(length(y0), 379);
for i = 1:379
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
c_f = 226;
figure(1)
pcolor(flip(reshape(X(:, c_f), [n m]))); shading interp, colormap(gray)
figure(2)
pcolor(flip(u_dmd(:,:,c_f))); shading interp, colormap(gray)
figure(3)
pcolor(flip(X_fg(:,:,c_f))); shading interp, colormap(gray)