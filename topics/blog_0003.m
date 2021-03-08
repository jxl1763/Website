%Make input images
I = imread("cameraman.tif");
I = double(I); % change I from unit8 to double precision format
% normalize I to intensity [0,1] for easier parameter selection
I = (I-min(I(:)))/(max(I(:))-min(I(:)));
[m,n] = size(I);
%% CASE 1
%Inpainting with missing pixels
%amount of removed pixels, 80%
perc = .8;
% random mask, Mask==1 for removed pixels
Mask = zeros(m,n);
Pick = randperm(m*n); 
Pick = Pick(1:round(perc*m*n));
Mask(Pick) = 1;
u0 = I;
u0(Mask == 1) = 0;
%display raw image
figure();
imshow(u0);
%pad boundary as the same elememts on boundary
u_pad = padarray(u0,[1 1],0,'both');
u_pad(:,1) = u_pad(:,2);
u_pad(:,258) = u_pad(:,257);
u_pad(1,:) = u_pad(2,:);
u_pad(258,:) = u_pad(257,:);

%setting
lambda = 1e4;
tol = 1e-4;
dt = 1e-4;
diff = 1;
u = u_pad;
count = 0;
[dim1, dim2] = size(u0);
u_update = zeros(dim1,dim2); %initialize
h = 1/dim1;
%Tikhonov method

while diff > tol
    lap = (u(3:dim1+2,2:dim2+1) + u(1:dim1,2:dim2+1) + u(2:dim1+1,3:dim2+2) + u(2:dim1+1,1:dim2) - 4 * u(2:dim1+1,2:dim2+1))/h^2;
    ut = -(- 2* lap / lambda); %for inpainting
    u_update(2:dim1+1, 2:dim2+1) = u(2:dim1+1, 2:dim2+1) + dt * ut;
    %or used the developed function
    %ut = -(u-u_pad -2*4*del2(u)/(lambda* h^2)); # for noise
    u_update(:,1) = u_update(:,2);
    u_update(:,dim2+2) = u_update(:,dim2+1);
    u_update(1,:) = u_update(2,:);
    u_update(dim1+2,:) = u_update(dim1+1,:);
    count = count+1;
    % for inpainting, project back, no need if doing noise remove
    pu = u_update(2:dim1+1,2:dim2+1);
    pu(Mask ~= 1) = u0(Mask ~= 1);
    u_update(2:dim1+1,2:dim2+1) = pu;
    diff = norm(u_update(2:dim1+1, 2:dim2+1) - u(2:dim1+1, 2:dim2+1),"fro");
    u = u_update;
    if mod(count,100) == 0
        diff
        imshow(u(2:dim1+1,2:dim2+1));
        pause(0.5);
    end
end

figure()
imshow(u(2:257,2:257));
psnr(u(2:257,2:257),I)
%10*log10(max(I,[],'all')^2 / mse(I - u(2:257,2:257))), the result is the
%same
norm(I-u(2:257,2:257),'fro') / norm(I, 'fro')

%TV
%setting
diff = 1; %to enter the loop
disp = 1e-6; % small value to add
u_update = zeros(dim1,dim2); %initialize
u = u_pad;
dt = 0.5*1e-5; %1e-5 work
tol = 0.045;
while diff > tol
    % center is area 2:dim1+1, 2:dim2+1
    dxp = u(3:dim1+2,2:dim2+1) - u(2:dim1+1,2:dim2+1); 
    dxm = u(2:dim1+1,2:dim2+1) - u(1:dim1, 2:dim2+1);
    dyp = u(2:dim1+1,3:dim2+2) - u(2:dim1+1,2:dim2+1);
    dym = u(2:dim1+1,2:dim2+1) - u(2:dim1+1,1:dim2);
    %center is area 1:dim1, 2:dim2+1, for calculating outside dxm
    dxmdxp = u(2:dim1+1, 2:dim2+1) - u(1:dim1,2:dim2+1);
    dxmdyp = u(1:dim1,3:dim2+2) - u(1:dim1,2:dim2+1);
    dxmdym = u(1:dim1,2:dim2+1) - u(1:dim1,1:dim2);
    %center is area 2:dim1+1, 1:dim2, for calculating outside dym
    dymdyp = u(2:dim1+1,2:dim2+1) - u(2:dim1+1,1:dim2);
    dymdxp = u(3:dim1+2,1:dim2) - u(2:dim1+1,1:dim2);
    dymdxm = u(2:dim1+1,1:dim2) - u(1:dim1,1:dim2);
    
    DXp = dxp./(disp + (dxp.^2 + ((sign(dyp)+sign(dym))*min(abs(dyp),abs(dym))./2.).^2).^0.5);
    DXm = dxmdxp./(disp + (dxmdxp.^2 + ((sign(dxmdyp) + sign(dxmdym)) * min(abs(dxmdyp),abs(dxmdym))./2).^2).^0.5);
    DYp = dyp./(disp + (dyp.^2+ ((sign(dxp)+sign(dxm))*min(abs(dxm),abs(dxp))./2).^2).^0.5);
    DYm = dymdyp./(disp + (dymdyp.^2 + ((sign(dymdxp) + sign(dymdxm)) * min(abs(dymdxp),abs(dymdxm))./2).^2).^0.5);
    
    %for inpainting, we don't have the term u - u0, then lambda is gone
    % maybe can times with dt, but then it works the same as adjust dt
    % directly
    
    u_update(2:dim1+1, 2:dim2+1) = u(2:dim1+1, 2:dim2+1) + dt*(DXp -DXm + DYp - DYm)/h;
    %u_update  = u + dt*(Xp -Xm + Yp - Ym)/h - dt*lambda*(u - u_pad); %for denoising 
    u_update(:,1) = u_update(:,2);
    u_update(:,dim2+2) = u_update(:,dim2+1);
    u_update(1,:) = u_update(2,:);
    u_update(dim1+2,:) = u_update(dim1+1,:);
    count = count+1;
    %project back
    pu = u_update(2:dim1+1,2:dim2+1);
    pu(Mask ~= 1) = u0(Mask ~= 1);
    u_update(2:dim1+1,2:dim2+1) = pu;
    diff = norm(u_update(2:dim1+1, 2:dim2+1) - u(2:dim1+1, 2:dim2+1),"fro");
    u = u_update;
    if mod(count,100) == 0
        diff
        imshow(u(2:dim1+1,2:dim2+1));
        pause(0.5);
    end
end

figure()
imshow(u(2:257,2:257));
psnr(u(2:257,2:257),I)
norm(I-u(2:257,2:257),'fro') / norm(I, 'fro')

%%
%CASE 2: denoising

sigma = 0.1;
J = I + sigma *randn(size(I));
imshow(J);
%pad J
u_pad = padarray(J,[1 1],0,'both');
u_pad(:,1) = u_pad(:,2);
u_pad(:,258) = u_pad(:,257);
u_pad(1,:) = u_pad(2,:);
u_pad(258,:) = u_pad(257,:);

%setting
lambda = 1e5;
tol = 1e-2;
dt = 1e-6;
diff = 1;
u = u_pad;
count = 0;
[dim1, dim2] = size(J);
u_update = zeros(dim1,dim2); %initialize
h = 1/dim1;
%Tikhonov method

while diff > tol
    lap = (u(3:dim1+2,2:dim2+1) + u(1:dim1,2:dim2+1) + u(2:dim1+1,3:dim2+2) + u(2:dim1+1,1:dim2) - 4 * u(2:dim1+1,2:dim2+1))/h^2;
    ut = -(- 2*lap); %for inpainting, modified lambda position to be inconsistance with TV model
    u_diff = u(2:dim1+1, 2:dim2+1) - u_pad(2:dim1+1, 2:dim2+1);
    ut = ut - lambda* u_diff; % move lambda to here
    u_update(2:dim1+1, 2:dim2+1) = u(2:dim1+1, 2:dim2+1) + dt * ut;
    u_update(:,1) = u_update(:,2);
    u_update(:,dim2+2) = u_update(:,dim2+1);
    u_update(1,:) = u_update(2,:);
    u_update(dim1+2,:) = u_update(dim1+1,:);
    count = count+1;
    diff = norm(u_update(2:dim1+1, 2:dim2+1) - u(2:dim1+1, 2:dim2+1),"fro");
    u = u_update;
    if mod(count,100) == 0
        diff
        imshow(u(2:dim1+1,2:dim2+1));
        pause(0.5);
    end
end

figure()
imshow(u(2:257,2:257));
psnr(u(2:257,2:257),I)
norm(I-u(2:257,2:257),'fro') / norm(I, 'fro')

%TV 

%TV
%setting
disp = 1e-6; % small value to add
u_update = zeros(dim1,dim2); %initialize
u = u_pad;
dt = 1e-5; %1e-5 work
max_iter = 1000; %TV is more easy to control using iteration times
lambda = 1e3;
count = 0;
while count < max_iter
    % center is area 2:dim1+1, 2:dim2+1
    dxp = u(3:dim1+2,2:dim2+1) - u(2:dim1+1,2:dim2+1); 
    dxm = u(2:dim1+1,2:dim2+1) - u(1:dim1, 2:dim2+1);
    dyp = u(2:dim1+1,3:dim2+2) - u(2:dim1+1,2:dim2+1);
    dym = u(2:dim1+1,2:dim2+1) - u(2:dim1+1,1:dim2);
    %center is area 1:dim1, 2:dim2+1, for calculating outside dxm
    dxmdxp = u(2:dim1+1, 2:dim2+1) - u(1:dim1,2:dim2+1);
    dxmdyp = u(1:dim1,3:dim2+2) - u(1:dim1,2:dim2+1);
    dxmdym = u(1:dim1,2:dim2+1) - u(1:dim1,1:dim2);
    %center is area 2:dim1+1, 1:dim2, for calculating outside dym
    dymdyp = u(2:dim1+1,2:dim2+1) - u(2:dim1+1,1:dim2);
    dymdxp = u(3:dim1+2,1:dim2) - u(2:dim1+1,1:dim2);
    dymdxm = u(2:dim1+1,1:dim2) - u(1:dim1,1:dim2);
    
    DXp = dxp./(disp + (dxp.^2 + ((sign(dyp)+sign(dym))*min(abs(dyp),abs(dym))./2.).^2).^0.5);
    DXm = dxmdxp./(disp + (dxmdxp.^2 + ((sign(dxmdyp) + sign(dxmdym)) * min(abs(dxmdyp),abs(dxmdym))./2).^2).^0.5);
    DYp = dyp./(disp + (dyp.^2+ ((sign(dxp)+sign(dxm))*min(abs(dxm),abs(dxp))./2).^2).^0.5);
    DYm = dymdyp./(disp + (dymdyp.^2 + ((sign(dymdxp) + sign(dymdxm)) * min(abs(dymdxp),abs(dymdxm))./2).^2).^0.5);
    
    u_update(2:dim1+1, 2:dim2+1) = u(2:dim1+1, 2:dim2+1) + dt*(DXp -DXm + DYp - DYm)/h;
    %further update for the scratch area
    u_diff = u(2:dim1+1, 2:dim2+1) - u_pad(2:dim1+1, 2:dim2+1);
    u_update_fur = u_update(2:dim1+1, 2:dim2+1);
    u_update_fur = u_update_fur - dt*lambda*u_diff; 
    u_update(2:dim1+1,2:dim2+1) = u_update_fur;
    %u_update  = u + dt*(Xp -Xm + Yp - Ym)/h - dt*lambda*(u - u_pad); %for denoising 
    u_update(:,1) = u_update(:,2);
    u_update(:,dim2+2) = u_update(:,dim2+1);
    u_update(1,:) = u_update(2,:);
    u_update(dim1+2,:) = u_update(dim1+1,:);
    count = count+1;
    diff = norm(u_update(2:dim1+1, 2:dim2+1) - u(2:dim1+1, 2:dim2+1),"fro");
    u = u_update;
    if mod(count,100) == 0
        diff
        imshow(u(2:dim1+1,2:dim2+1));
        pause(0.5);
    end
end

figure()
imshow(u(2:257,2:257));
psnr(u(2:257,2:257),I)
norm(I-u(2:257,2:257),'fro') / norm(I, 'fro')


%%
%CASE 3: denoising and inpainting
%Test on image with scratch
load("../data/scratch.mat");
J = I;
J(scratch == 0) = 0;
imshow(J);
sigma = 0.1;
J = J + sigma *randn(size(J));
imshow(J);
%pad J
u_pad = padarray(J,[1 1],0,'both');
u_pad(:,1) = u_pad(:,2);
u_pad(:,258) = u_pad(:,257);
u_pad(1,:) = u_pad(2,:);
u_pad(258,:) = u_pad(257,:);

%setting
lambda = 1e5;
tol = 1e-2;
dt = 1e-6;
diff = 1;
u = u_pad;
count = 0;
[dim1, dim2] = size(J);
u_update = zeros(dim1,dim2); %initialize
h = 1/dim1;
%Tikhonov method

while diff > tol
    lap = (u(3:dim1+2,2:dim2+1) + u(1:dim1,2:dim2+1) + u(2:dim1+1,3:dim2+2) + u(2:dim1+1,1:dim2) - 4 * u(2:dim1+1,2:dim2+1))/h^2;
    ut = -(- 2*lap); %for inpainting, modified lambda position to be inconsistance with TV model
    u_diff = u(2:dim1+1, 2:dim2+1) - u_pad(2:dim1+1, 2:dim2+1);
    ut(scratch == 1) = ut(scratch == 1) - lambda* u_diff(scratch == 1); % move lambda to here
    u_update(2:dim1+1, 2:dim2+1) = u(2:dim1+1, 2:dim2+1) + dt * ut;
    u_update(:,1) = u_update(:,2);
    u_update(:,dim2+2) = u_update(:,dim2+1);
    u_update(1,:) = u_update(2,:);
    u_update(dim1+2,:) = u_update(dim1+1,:);
    count = count+1;
    diff = norm(u_update(2:dim1+1, 2:dim2+1) - u(2:dim1+1, 2:dim2+1),"fro");
    u = u_update;
    if mod(count,100) == 0
        diff
        imshow(u(2:dim1+1,2:dim2+1));
        pause(0.5);
    end
end

figure()
imshow(u(2:257,2:257));
psnr(u(2:257,2:257),I)
norm(I-u(2:257,2:257),'fro') / norm(I, 'fro')

%TV 

%TV
%setting
diff = 1; %to enter the loop
disp = 1e-6; % small value to add
u_update = zeros(dim1,dim2); %initialize
u = u_pad;
dt = 1e-5; %1e-5 work
max_iter = 1000; %TV is more easy to control using iteration times
lambda = 1e3;
count = 0;
while count < max_iter
    % center is area 2:dim1+1, 2:dim2+1
    dxp = u(3:dim1+2,2:dim2+1) - u(2:dim1+1,2:dim2+1); 
    dxm = u(2:dim1+1,2:dim2+1) - u(1:dim1, 2:dim2+1);
    dyp = u(2:dim1+1,3:dim2+2) - u(2:dim1+1,2:dim2+1);
    dym = u(2:dim1+1,2:dim2+1) - u(2:dim1+1,1:dim2);
    %center is area 1:dim1, 2:dim2+1, for calculating outside dxm
    dxmdxp = u(2:dim1+1, 2:dim2+1) - u(1:dim1,2:dim2+1);
    dxmdyp = u(1:dim1,3:dim2+2) - u(1:dim1,2:dim2+1);
    dxmdym = u(1:dim1,2:dim2+1) - u(1:dim1,1:dim2);
    %center is area 2:dim1+1, 1:dim2, for calculating outside dym
    dymdyp = u(2:dim1+1,2:dim2+1) - u(2:dim1+1,1:dim2);
    dymdxp = u(3:dim1+2,1:dim2) - u(2:dim1+1,1:dim2);
    dymdxm = u(2:dim1+1,1:dim2) - u(1:dim1,1:dim2);
    
    DXp = dxp./(disp + (dxp.^2 + ((sign(dyp)+sign(dym))*min(abs(dyp),abs(dym))./2.).^2).^0.5);
    DXm = dxmdxp./(disp + (dxmdxp.^2 + ((sign(dxmdyp) + sign(dxmdym)) * min(abs(dxmdyp),abs(dxmdym))./2).^2).^0.5);
    DYp = dyp./(disp + (dyp.^2+ ((sign(dxp)+sign(dxm))*min(abs(dxm),abs(dxp))./2).^2).^0.5);
    DYm = dymdyp./(disp + (dymdyp.^2 + ((sign(dymdxp) + sign(dymdxm)) * min(abs(dymdxp),abs(dymdxm))./2).^2).^0.5);
    
    u_update(2:dim1+1, 2:dim2+1) = u(2:dim1+1, 2:dim2+1) + dt*(DXp -DXm + DYp - DYm)/h;
    %further update for the scratch area
    u_diff = u(2:dim1+1, 2:dim2+1) - u_pad(2:dim1+1, 2:dim2+1);
    u_update_fur = u_update(2:dim1+1, 2:dim2+1);
    u_update_fur(scratch == 1) = u_update_fur(scratch == 1) - dt*lambda*u_diff(scratch == 1); 
    u_update(2:dim1+1,2:dim2+1) = u_update_fur;
    %u_update  = u + dt*(Xp -Xm + Yp - Ym)/h - dt*lambda*(u - u_pad); %for denoising 
    u_update(:,1) = u_update(:,2);
    u_update(:,dim2+2) = u_update(:,dim2+1);
    u_update(1,:) = u_update(2,:);
    u_update(dim1+2,:) = u_update(dim1+1,:);
    count = count+1;
    %project back
    diff = norm(u_update(2:dim1+1, 2:dim2+1) - u(2:dim1+1, 2:dim2+1),"fro");
    u = u_update;
    if mod(count,100) == 0
        diff
        imshow(u(2:dim1+1,2:dim2+1));
        pause(0.5);
    end
end

figure()
imshow(u(2:257,2:257));
psnr(u(2:257,2:257),I)
norm(I-u(2:257,2:257),'fro') / norm(I, 'fro')
