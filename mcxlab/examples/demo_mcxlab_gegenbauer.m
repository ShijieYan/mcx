%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% MCXLAB - Monte Carlo eXtreme for MATLAB/Octave by Qianqina Fang
%
% In this example, we show how to use the two-term Gegenbauer scattering
% function in MCX.
%
% This file is part of Monte Carlo eXtreme (MCX) URL:http://mcx.sf.net
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear
% photon number
cfg.nphoton=1000000;

% light source
cfg.srcpos=[30 30 0];
cfg.srcdir=[0 0 1];

% gpu settings
cfg.gpuid=1;
cfg.autopilot=1;

% label-based segmented volume
cfg.vol=uint8(ones(60,60,60));

% optical properties mua, mus, gf, n
cfg.prop=[0 0 1 1;           % label 0
          0.005 1 0.8 1.37]; % label 1

% add parameters for two-term gegenbauer scattering: af, gb, ab, c
cfg.gegenprop=[0.5 0.8 0.5 1]; % label 1

% boundary condition
cfg.bc='______001000'; % get reflected photons

% output photon profile: scattering count[s], partial path-lengths[p], exit
% direction[v], initial weight[w]
cfg.savedetflag='spvw';

% time gate settings
cfg.tstart=0;
cfg.tend=5e-9;
cfg.tstep=5e-9;

% run simulations
[flux,detp]=mcxlab(cfg);