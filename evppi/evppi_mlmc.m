%
% Code written by Takashi Goda with minor changes by Mike Giles
%
% Based on testcase by Brennan, Kharroubi, O'Hagan, Chilcott in
% "Calculating partial expected value of perfect information via
%  Monte Carlo sampling algorithms", Medical Decision Making. 
% 27(4):448-70, 2007.
%

function test1

close all; clear all;

%
% MLMC treatment
%

addpath('..');

N      = 100000;    % samples for convergence tests
L      = 10;        % levels for convergence tests 

N0     = 1000;   % initial samples on coarse levels
Lmin   = 2;      % minimum refinement level
Lmax   = 20;     % maximum refinement level
% Eps    = [ 0.1 0.2 0.5 1 2];
Eps    = [ 0.5 1 2 5 10];

filename = 'test1';
fp       = fopen([filename '.txt'],'w');
mlmc_test(@test1_l, N,L, N0,Eps,Lmin,Lmax, fp);
fclose(fp);

%
% plot results
%

nvert = 3;
mlmc_plot(filename, nvert);

if(nvert==1)
  figure(1)
  print('-deps2',[filename 'a.eps'])
  figure(2)
  print('-deps2',[filename 'b.eps'])
else
  print('-deps2',[filename '.eps'])
end

%-------------------------------------------------------
%
% level l estimator
%

function [sums cost] = test1_l(l,N)

% global option

nf = 2^l;
nc = nf/2;

sums(1:6) = 0;

len = floor(1000000/nf);  % strip-mining to limit memory requirement

for N1 = 1:len:N
  N2 = min(len,N-N1+1);

% level 0

  if (l==0)
    Pf = zeros(1,N2);
    dP = Pf;

  else

% level l>0, with antithetic sampler

    lmd = 10000;
    rho1 = 0.6;
    rho2 = 0.0;

%   outer loop
    mu = [0.7,3,0.8,3];
    r5 = 0.1;
    r7 = 0.5;
    r14 = 0.1;
    r16 = 1;
    rh = rho1;
    sgm = [r5^2,rh*r5*r7,rh*r5*r14,rh*r5*r16; ... 
           rh*r5*r7,r7^2,rh*r7*r14,rh*r7*r16; ...
           rh*r5*r14,rh*r7*r14,r14^2,rh*r14*r16; ...
           rh*r5*r16,rh*r7*r16,rh*r14*r16,r16^2] ;
    Z = mvnrnd(mu,sgm,N2);
    X5 = Z(:,1).';
    X14 = Z(:,3).';

%   inner loop
    X1a = normrnd(1000,1,nc,N2);
    X1b = normrnd(1000,1,nc,N2);
    X2a = normrnd(0.1,0.02,nc,N2);
    X2b = normrnd(0.1,0.02,nc,N2);
    X3a = normrnd(5.2,1,nc,N2);
    X3b = normrnd(5.2,1,nc,N2);
    X4a = normrnd(400,200,nc,N2);
    X4b = normrnd(400,200,nc,N2);
    X8a = normrnd(0.25,0.1,nc,N2);
    X8b = normrnd(0.25,0.1,nc,N2);
    X9a = normrnd(-0.1,0.02,nc,N2);
    X9b = normrnd(-0.1,0.02,nc,N2);
    X10a = normrnd(0.5,0.2,nc,N2);
    X10b = normrnd(0.5,0.2,nc,N2);
    X11a = normrnd(1500,1,nc,N2);
    X11b = normrnd(1500,1,nc,N2);
    X12a = normrnd(0.08,0.02,nc,N2);
    X12b = normrnd(0.08,0.02,nc,N2);
    X13a = normrnd(6.1,1,nc,N2);
    X13b = normrnd(6.1,1,nc,N2);
    X17a = normrnd(0.2,0.05,nc,N2);
    X17b = normrnd(0.2,0.05,nc,N2);
    X18a = normrnd(-0.1,0.02,nc,N2);
    X18b = normrnd(-0.1,0.02,nc,N2);
    X19a = normrnd(0.5,0.2,nc,N2);
    X19b = normrnd(0.5,0.2,nc,N2);

    X6a = zeros(nc,N2);
    X6b = zeros(nc,N2);
    X15a = zeros(nc,N2);
    X15b = zeros(nc,N2);
    mu = [0.3,0.3];
    r6 = 0.1;
    r15 = 0.05;
    rh = rho2;
    sgm = [r6^2,rh*r6*r15;rh*r6*r15,r15^2];
    Za = mvnrnd(mu,sgm,nc*N2);
    Zb = mvnrnd(mu,sgm,nc*N2);
    X6a = reshape(Za(:,1), nc, N2);
    X6b = reshape(Zb(:,1), nc, N2);
    X15a = reshape(Za(:,2), nc, N2);
    X15b = reshape(Zb(:,2), nc, N2);

    X7a = zeros(nc,N2);
    X7b = zeros(nc,N2);
    X16a = zeros(nc,N2);
    X16b = zeros(nc,N2);
    r5 = 0.1;
    r7 = 0.5;
    r14 = 0.1;
    r16 = 1;
    rh = rho1;
    mu1 = [3,3];
    mu2 = [0.7,0.8];
    sgm11 = [r7^2,rh*r7*r16;rh*r7*r16,r16^2];
    sgm12 = [rh*r7*r5,rh*r7*r14;rh*r16*r5,rh*r16*r14];
    sgm21 = [rh*r5*r7,rh*r5*r16;rh*r14*r7,rh*r14*r16];
    sgm22 = [r5^2,rh*r5*r14;rh*r14*r5,r14^2];
    sgm = sgm11-sgm12*inv(sgm22)*sgm21;
    mu1 = repmat(mu1,N2,1);
    mu2 = repmat(mu2,N2,1);
    mu = mu1 + (sgm12*inv(sgm22)*([X5; X14].'-mu2).').';
    Za = repmat(mu,nc,1) + mvnrnd([0 0],sgm,N2*nc);
    Zb = repmat(mu,nc,1) + mvnrnd([0 0],sgm,N2*nc);
    X7a = reshape(Za(:,1), N2, nc).';
    X7b = reshape(Zb(:,1), N2, nc).';
    X16a = reshape(Za(:,2), N2, nc).';
    X16b = reshape(Zb(:,2), N2, nc).';


    Pca1 = lmd*(X5.*mean(X6a.*X7a,1)+mean(X8a.*X9a.*X10a,1)) - (mean(X1a,1)+mean(X2a.*X3a.*X4a,1));
    Pca2 = lmd*(X14.*mean(X15a.*X16a,1)+mean(X17a.*X18a.*X19a,1)) - (mean(X11a,1)+mean(X12a.*X13a.*X4a,1));
    Pcb1 = lmd*(X5.*mean(X6b.*X7b,1)+mean(X8b.*X9b.*X10b,1)) - (mean(X1b,1)+mean(X2b.*X3b.*X4b,1));
    Pcb2 = lmd*(X14.*mean(X15b.*X16b,1)+mean(X17b.*X18b.*X19b,1)) - (mean(X11b,1)+mean(X12b.*X13b.*X4b,1));
    Pf1 = (Pca1+Pcb1)/2;
    Pf2 = (Pca2+Pcb2)/2;
    Pf = max(Pf1,Pf2);
    Pc = (max(Pca1,Pca2)+max(Pcb1,Pcb2))/2;
    dP = Pc-Pf;

    X5 = repmat(X5,nc,1);
    X14 = repmat(X14,nc,1);
    P1a1 = lmd*(X5.*X6a.*X7a+X8a.*X9a.*X10a) - (X1a+X2a.*X3a.*X4a);
    P1a2 = lmd*(X14.*X15a.*X16a+X17a.*X18a.*X19a) - (X11a+X12a.*X13a.*X4a);
    P1b1 = lmd*(X5.*X6b.*X7b+X8b.*X9b.*X10b) - (X1b+X2b.*X3b.*X4b);
    P1b2 = lmd*(X14.*X15b.*X16b+X17b.*X18b.*X19b) - (X11b+X12b.*X13b.*X4b);
    P1 = sum(max(P1a1,P1a2)+max(P1b1,P1b2),1)/nf;
    Pf = P1-Pf;
  end

  sums(1) = sums(1) + sum(dP);
  sums(2) = sums(2) + sum(dP.^2);
  sums(3) = sums(3) + sum(dP.^3);
  sums(4) = sums(4) + sum(dP.^4);
  sums(5) = sums(5) + sum(Pf);
  sums(6) = sums(6) + sum(Pf.^2);
end

cost = N*nf;


