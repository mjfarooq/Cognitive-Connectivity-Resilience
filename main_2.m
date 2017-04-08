
clc; clear all; close all;
rng(235);

kappa = 1.2;
d = 20; % flock distance or scale 
r = kappa * d; % communication range of devices;
Maxiter = 2000;
Num_drones = 70; % Maximum number of overlay devices
Num_users_max = 80; % maximum number of users supported by each device


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

q = normrnd(0, 50,Num_drones,2); % Coordinates of points
%q = [-120*ones(Num_drones,1) -80*ones(Num_drones,1)];
p = -unifrnd(1,2, length(q),2); % Initial velocity of agents



%------------------------------------------------------------------------------------
%               Generation of Layer 1 Devices and Centroids
%------------------------------------------------------------------------------------
Mu = [30 40;-20 -20; -80 60];
Sigma = cat(3,[200 0;0 100],[500 0;0 200], [150 0;0 300]);
P = ones(1,3)/3;
gm = gmdistribution(Mu,Sigma,P);

Y = random(gm,2000);
figure;
scatter(Y(:,1),Y(:,2),10,'.')
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

scatter(q(:,1), q(:,2))
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
Ts = 0.01; % Sampling time for discrete system
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

for iter = 1:Maxiter
    iter
    X_old = X_new;
    for i = 1:length(q)
        X_new(i,:) = sysd.A * X_old(i,:)' + sysd.B * node(i).input;  % X_dot = A_d*X + B_d * U where X = [q1 q2 p1 p2]' and U = [u1 u2]' 
    end
    
    dt = delaunayTriangulation(X_new(:,1:2));
    [xi, D] = nearestNeighbor(dt, Y);
    xi(D > r) = 0;

    for i = 1:length(q)
        Num_users(i) = sum(xi == i);
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        A_new = zeros(length(q),length(q)); % Adjacency matrix of graph
        for ii = 1: size(A_new,1)
            for jj = 1:size(A_new,2)
                if ii ~= jj
                    A_new(ii,jj) = bump(   sigma_norm(X_new(jj,1:2) - X_new(ii,1:2)) / sigma_norm(r)   ,h);
                end
            end
        end
        
        
        for m = 1:length(q) % Re-evaluating neighbours
            node(m).neighbours = find(A_new(m,:) ~= 0);
        end 
        
        for j = 1:length(q)
            gradient_term = zeros(2,1); consensus_term = zeros(2,1);
            for k = 1: length(node(j).neighbours)
                    gradient_term  = gradient_term + (     phi_alpha( sigma_norm(X_new(node(j).neighbours(k),1:2) - X_new(j,1:2) ))   +    4.9*(1 - bump(         ( max( 0, Num_users(node(j).neighbours(k)) - Num_users_max))/5, 0 ))     ) * sigma_grad(X_new(node(j).neighbours(k),1:2)  - X_new(j,1:2))'  ; % -        
                    consensus_term = consensus_term + A_new(j,node(j).neighbours(k)) * (X_new(node(j).neighbours(k),3:4) - X_new(j,3:4))';
            end
            node(j).input =  gradient_term + consensus_term - 0.1*(X_new(j,1:2)' - C(nearest_cluster(X_new(j,1:2),C),:)') - 0.1*(X_new(j,3:4)' - 0);
        end % Re-evauating input
        
        over_capacity(iter) = sum(max(0, Num_users - Num_users_max));
        unallocated(iter) = sum(xi == 0);
        Total_unserved_users(iter) = over_capacity(iter) + unallocated(iter);
        
        for i = 1:length(q)
            deg(i) = sum(A_new(i,:) > 0);
            I(i) = 1 - 1/(1 + tau*deg(i)); % upper bound for infection probability
        end
        
        I_avg(iter) = mean(I); % average infection probability;
        Capacity_utilization(iter) = (length(q) * Num_users_max - sum(min(Num_users, Num_users_max))) / (length(q) * Num_users_max);
        Percent_users_covered(iter) = (length(q)*Num_users_max - Total_unserved_users(iter))/ (length(q)*Num_users_max); 
%         if mod(iter,100) == 0
%             figure
%             scatter(X_new(:,1), X_new(:,2))
%             hold on
%         end
        
end




figure
scatter(Y(:,1),Y(:,2),10,'.')
title('Final Configuration');
hold on
scatter(X_new(:,1), X_new(:,2))
text(X_new(:,1), X_new(:,2) , num2str(Num_users))
gplot(A_new,X_new(:,1:2),'-o')


figure
plot(1:Maxiter, Total_unserved_users)
ylabel('Total Unserved Users')
xlabel('Time')


figure
plot(1:Maxiter, I_avg)
ylabel('Reachability of Overlay Network')
xlabel('Time')

figure
plot(1:Maxiter, Capacity_utilization)
ylabel('Capacity utilization of overlay network')
xlabel('Time')

figure
plot(1:Maxiter, Percent_users_covered)
ylabel('Capacity utilization of overlay network')
xlabel('Time')










