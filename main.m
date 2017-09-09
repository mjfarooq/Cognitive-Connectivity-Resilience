clc; clear all; close all

v = VideoWriter('uav_without_failure_flat_new2.avi');
v.FrameRate = 8;  % Default 30
v.Quality = 100;    % Default 75
open(v);

rng(457);

kappa = 1.2;
d = 20; % flock distance or scale 
r = kappa * d; % communication range of devices;
Maxiter = 900; % Maximum number of iterations
Plotiter = 1; % Iterations after which video recording starts
Attack_iter = 450; % Iterations after which a random attack is induced
Num_drones = 80; % Maximum number of overlay devices
Num_users_max = 80; % maximum number of users supported by each device
Num_mobile_devices = 2000; % Total number of ground users 

epsilon = 0.1; % parameter of sigma norm [0,1]
sigma_norm = @(x) (1/epsilon)*(sqrt(1 + epsilon*norm(x)^2) - 1);
sigma_grad = @(x) 1/(1 + epsilon*sigma_norm(x)) * x;
h = 0.2; % argument of bump function

sigma_1 = @(z)  z/(sqrt(1 + z^2));
a = 5;
b = 5;
c = abs(a - b)/sqrt(4*a*b); 
phi = @(z) 0.5*( (a + b)*sigma_1(z + c)   +  (a - b) );
phi_alpha = @(z) bump(z / sigma_norm(r)  , h) * phi(z - sigma_norm(d));

q = unifrnd(-100, 100,Num_drones,2); % Coordinates of points
%q = [-120*ones(Num_drones,1) -80*ones(Num_drones,1)];
p = -unifrnd(1,2, length(q),2); % Initial velocity of agents


%------------------------------------------------------------------------------------
%               Generation of Layer 1 Devices and Centroids
%------------------------------------------------------------------------------------
Mu = [30 40;-20 -20; -80 60];
Sigma = cat(3,[200 0;0 100],[500 0;0 200], [150 0;0 300]);
P = ones(1,3)/3;
gm = gmdistribution(Mu,Sigma,P);

Y = random(gm,Num_mobile_devices);
figure;
scatter3(Y(:,1),Y(:,2), zeros(Num_mobile_devices,1),10,'.')
%scatter(Y(:,1),Y(:,2),10,'.')
title('GMM - PDF Contours and Simulated Data');

k = 10; % Number of desired cluster centers
% [idx C] = kmedoids(Y,k); % Rows of C are the cluster centers
opts = statset('Display','final');
[idx,C] = kmeans(Y,k,'Distance','cityblock','Replicates',5,'Options',opts);

hold on;
plot(C(:,1),C(:,2),'kx','MarkerSize',15,'LineWidth',3)
%------------------------------------------------------------------------------------


A = zeros(length(q),length(q)); % Adjacency matrix of graph
for i = 1: size(A,1)
    for j = 1:size(A,2)
        if i ~= j
            A(i,j) = bump(   sigma_norm(q(i,:) - q(j,:)) / sigma_norm(r)   ,h);
        end
    end
end

scatter3(q(:,1),q(:,2), zeros(length(q(:,1)),1),11,'.')
%scatter(q(:,1), q(:,2))
hold on
gplot(A,q,'-o')


node = struct;

for i = 1:length(q) % List of neighbours
    node(i).neighbours = find(A(i,:) ~= 0);
end

dt = delaunayTriangulation(q);
[xi, D] = nearestNeighbor(dt, Y);
xi(D > r) = 0;

Num_users = zeros(length(q),1);
for i = 1:length(q)
    Num_users(i) = sum(xi == i);
end

%u = struct; 
for i = 1:length(q)
    gradient_term = zeros(2,1); consensus_term = zeros(2,1);
    for k = 1: length(node(i).neighbours)
            gradient_term  = gradient_term + (      phi_alpha( sigma_norm(q(node(i).neighbours(k),:) - q(i,:) ))     +     4.9*(1 - bump( ( max(0 , Num_users(node(i).neighbours(k)) - Num_users_max))/5, 0)) )* sigma_grad(q(node(i).neighbours(k),:)  - q(i,:))'   ; % -      
            consensus_term = consensus_term + A(i,node(i).neighbours(k)) * (p(node(i).neighbours(k),:) - p(i,:))';
    end
    node(i).input =  gradient_term + consensus_term - 0.1*(q(i,:)' - C(nearest_cluster(q(i,:),C),:)') - 0.1*(p(i,:)' - 0) ;         
end

mat_A = [0 0 1 0; 0 0 0 1; 0 0 0 0; 0 0 0 0];
mat_B = [0 0; 0 0; 1 0; 0 1];
sys = ss(mat_A, mat_B, [], []); % State space model
Ts = 0.05; % Sampling time for discrete system
sysd = c2d(sys,Ts); % Discrete state space model



q_old = q; % Old position
p_old = p; % Old velocity

%%%%%%%%%%%  Information Diffusion %%%%%%%%%%%%
beta = 1; 
delta = 1;
tau = beta / delta;

I = zeros(length(q),1); % Infection probability of each overlay device
deg = zeros(length(q),1); % degree of device i

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%




X_new = [q p];
over_capacity = zeros(1,Maxiter);
unallocated = zeros(1,Maxiter);
Total_unserved_users = zeros(1,Maxiter);
I_avg = zeros(1,Maxiter);
Capacity_utilization = zeros(1,Maxiter);
Percent_users_covered = zeros(1,Maxiter);
Failed_coord = [];
Fiedler = zeros(1,Maxiter);

figure;

for iter = 1:Maxiter
    iter
    X_old = X_new;
    for i = 1:length(X_new)
        X_new(i,:) = sysd.A * X_old(i,:)' + sysd.B * node(i).input;  % X_dot = A_d*X + B_d * U where X = [q1 q2 p1 p2]' and U = [u1 u2]' 
    end
    
    %%%%%%%%%%%%%%%%%%%%%%% Mobility of users %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % The mobility of users is modeled by a scaled 2-D random walk
    scale = 0.3; % Scale of Random Walk
    Y = Y + scale* unifrnd(-1,1,Num_mobile_devices,2);
    [idx,C] = kmeans(Y,k,'Distance','cityblock','Replicates',5,'Options',opts); % Re-evaluate Cluster Centers
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    dt = delaunayTriangulation(X_new(:,1:2));
    [xi, D] = nearestNeighbor(dt, Y);
    xi(D > r) = 0;

    Num_users = zeros(length(X_new),1);
    for i = 1:length(X_new)
        Num_users(i) = sum(xi == i);
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        deg = zeros(length(X_new),1); % degree of device i
        A_new = zeros(length(X_new),length(X_new)); % Adjacency matrix of graph
        I = zeros(length(X_new),1); % Infection probability of each overlay device

        for ii = 1: size(A_new,1)
            for jj = 1:size(A_new,2)
                if ii ~= jj
                    A_new(ii,jj) = bump(   sigma_norm(X_new(jj,1:2) - X_new(ii,1:2)) / sigma_norm(r)   ,h);
                end
            end
        end
        
        
        for m = 1:length(X_new) % Re-evaluating neighbours
            node(m).neighbours = find(A_new(m,:) ~= 0);
        end 
        
        for j = 1:length(X_new)
            gradient_term = zeros(2,1); consensus_term = zeros(2,1);
            for k = 1: length(node(j).neighbours)
                    gradient_term  = gradient_term + (     phi_alpha( sigma_norm(X_new(node(j).neighbours(k),1:2) - X_new(j,1:2) ))   +    4.9*(1 - bump(         ( max( 0, Num_users(node(j).neighbours(k)) - Num_users_max) )/5, 0 ))     ) * sigma_grad(X_new(node(j).neighbours(k),1:2)  - X_new(j,1:2))'  ; % -        
                    consensus_term = consensus_term + A_new(j,node(j).neighbours(k)) * (X_new(node(j).neighbours(k),3:4) - X_new(j,3:4))';
            end
            node(j).input =  gradient_term + consensus_term - 0.1*(X_new(j,1:2)' - C(nearest_cluster(X_new(j,1:2),C),:)') - 0.1*(X_new(j,3:4)' - 0);
        end % Re-evauating input
        
        over_capacity(iter) = sum(max(0, Num_users - Num_users_max));
        unallocated(iter) = sum(xi == 0);
        Total_unserved_users(iter) = over_capacity(iter) + unallocated(iter);
        
        for i = 1:length(X_new)
            deg(i) = sum(A_new(i,:) > 0);
            I(i) = 1 - 1/(1 + tau*deg(i)); % upper bound for infection probability
        end
        
        I_avg(iter) = mean(I); % average infection probability;
        Capacity_utilization(iter) = (length(X_new) * Num_users_max - sum(min(Num_users, Num_users_max))) / (length(X_new) * Num_users_max);
        Percent_users_covered(iter) = (length(X_new)*Num_users_max - Total_unserved_users(iter))/ (length(X_new)*Num_users_max); 
        
        D = diag(deg);
        Lap =  D - (A_new > 0);
        Eigenvalues = eig(Lap);
        Fiedler(iter) = Eigenvalues(2);
        
        
        clf
        
        
        
        sc1 = scatter3(Y(:,1),Y(:,2), zeros(Num_mobile_devices,1),12,'.')

        hold on
        sc2 = scatter3(X_new(:,1),X_new(:,2), 20*ones(length(X_new(:,1)),1),30,'^','Filled')
        if iter > Attack_iter
            %sc3 = scatter(Failed_coord(:,1), Failed_coord(:,2),'*','r');
            sc3 = scatter3(Failed_coord(:,1),Failed_coord(:,2), 20*ones(length(Failed_coord(:,1)),1),30,'*','black')
            legend([sc2,sc1,sc3],'Overlay Device','Underlay Device', 'Failed Overlay Device','location','northwest');
        else
            legend([sc2,sc1],'Overlay Device','Underlay Device','location','northwest');
        end
        

        [X_ax, Y_ax] = gplot(A_new,[X_new(:,1:2) 20*ones(length(X_new),1)]);
        plot3(X_ax,Y_ax, 20*ones(length(X_ax),1), 'Color', [0.9290 ,0.6940,0.1250])
        %gplot(A_new,X_new(:,1:2),'-')
        %axis([-150 150 -150 150]) 
        view(-55,43)
        %view(-180,90)  Uncomment to change viewing angle of plot
        axis([-150 100 -150 150 0 30])
        drawnow;
        if iter > Plotiter
        frame = getframe(gcf);
        writeVideo(v,frame);
        end
        %%%%%%%%%%%%%% Failure Introduced at 1000 iter %%%%%%%%%%%%%%%%%%%
        failure_prop = 0.40; % Percentage of devices failed
        if iter == Attack_iter
            failed_id = randsample(length(X_new), length(X_new)*failure_prop);
            Failed_coord = X_new(failed_id,:);
            X_new = X_new( setdiff(1:length(X_new),failed_id) ,:);
        end
               
end





close(v);
