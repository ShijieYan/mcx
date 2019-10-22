%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% MCXLAB - Monte Carlo eXtreme for MATLAB/Octave by Qianqina Fang
%
% In this example, we show how to define continuously varying media 
% (i.e. media optical properties vary from voxel to voxel)
%
% This file is part of Monte Carlo eXtreme (MCX) URL:http://mcx.sf.net
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear cfg cfgs
cfg.nphoton=1e8;
cfg.vol=uint8(ones(60,60,60));
cfg.srcpos=[30 30 1];
cfg.srcdir=[0 0 1];
cfg.gpuid=1;
% cfg.gpuid='11'; % use two GPUs together
cfg.autopilot=1;
cfg.prop=[0 0 1 1;0.005 1 0 1.37];
cfg.tstart=0;
cfg.tend=5e-9;
cfg.tstep=5e-9;
% calculate the flux distribution with the given config

%% Approach #1: a 1 x Nx x Ny x Nz floating-point array defines mua(x,y,z)
%define continuously varying media - use a 4D array - 1st dimension defines
%the optical property components of each spatial position

mua1=0.003;
mua2=0.1;
mua=single(reshape(repmat([mua1:(mua2-mua1)/(60-1):mua2]',60,60),60,60,60));
cfg.mua=mua; % make 1st dimension the property dimension

flux=mcxlab(cfg);

% mcxplotvol(squeeze(cfg.vol))
% title('continuously varying absorption coeff \mu_a (1/mm)')
% view([-25.5,21.2]);
% 
% mcxplotvol(log10(flux.data))
% colormap(jet);
% title('fluence in continuously varying media (1/mm^2)')
% view([-25.5,21.2]);