%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% MCXLAB - Monte Carlo eXtreme for MATLAB/Octave by Qianqina Fang
%
% In this example, we show how to build Jacobians for multi-pattern src-det 
% pairs in MCXLAB using photon sharing and replay.
%
% This file is part of Monte Carlo eXtreme (MCX) URL:http://mcx.sf.net
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% prepare srcpattern and detpattern for photon sharing and multi-wide_det replay
addpath('../'); %addpath to auxiliary functions under 'path_to_mcx/mcx/mcxlab/' 
clear srcpattern detpattern

srcpattern=zeros(4,2,2);     % in this example, use a 2x2 pattern
    srcpattern(1,:,:)=[1,1;0,0]; % half illumination for each pattern
    srcpattern(2,:,:)=[0,0;1,1];
    srcpattern(3,:,:)=[0,1;0,1];
    srcpattern(4,:,:)=[1,0;1,0];
    srcparam1=[40 0 0 size(srcpattern,2)];
    srcparam2=[0 40 0 size(srcpattern,3)];

detpattern=zeros(4,2,2);     % for pattern detection, also use a 2x2 pattern
    detpattern(1,:,:)=[1,0;0,0]; % one pixel each time
    detpattern(2,:,:)=[0,1;0,0];
    detpattern(3,:,:)=[0,0;1,0];
    detpattern(4,:,:)=[0,0;0,1];
    detpos=[10 10 60];
    detparam1=[40 0 0 size(detpattern,2)];
    detparam2=[0 40 0 size(detpattern,3)];
    detorientation='+z'; %orientation of pattern detector, for now 6 directions are supported

%% forward simulation for fully illuminated src and det pattern
clear cfg

cfg.nphoton=3e8;
cfg.vol=uint8(ones(60,60,60));
cfg.srctype='pattern';
cfg.srcnum=1; %for base simulation, always use 1 srcpattern
cfg.srcpos=[10 10 0];
cfg.issrcfrom0=1; %necessary flag
cfg.srcdir=[0 0 1];
cfg.srcparam1=srcparam1;
cfg.srcparam2=srcparam2;
% srcnum>1, use ones as the base simulation; srcunm=1, use itself for the
%base simulation
cfg.srcpattern=ones(cfg.srcnum,cfg.srcparam1(4),cfg.srcparam2(4));
cfg.gpuid=4;
cfg.autopilot=1;
cfg.prop=[0 0 1 1;0.005 1 0 1.37];
cfg.tstart=0;
cfg.tend=5e-9;
cfg.tstep=5e-9;
cfg.dettype='pattern'; %internally mcx do not support rectangular-shaped pattern detectors, additional pre- and post-processing is necessary
    cfg.detpos=detpos;
    cfg.detparam1=detparam1;
    cfg.detparam2=detparam2;
    cfg.detorientation='+z'; 
cfg=mcxdetpattern(cfg); %pre-processing: convert pattern detector to sphere detector
cfg.issaveexit=1; %retrieve photon exiting position and direction for replay preprocessing
cfg.unitinmm=1;
% calculate the flux distribution with the given config
[flux,detp,vol,seeds]=mcxlab(cfg);

%% post-processing to sort out the detected photons that should be captured by the user-defined detection patterns
% to use wide-field pattern detection, the simulation flag srcfrom0 must be 1
[replaydetidx,newdetpdata,newseeds]=mcxdetpidx(detpos,detparam1,detparam2,detorientation,detp,seeds,cfg.unitinmm);

%% jacobian for multi src and det patterns
newcfg=cfg;
newcfg.srcpattern=srcpattern;
newcfg.srcnum=size(newcfg.srcpattern,1);
newcfg.detpattern=detpattern;
newcfg.seed=newseeds;
newcfg.outputtype='jacobian';
newcfg.detphotons=newdetpdata;
newcfg.replaydetidx=replaydetidx;

%% run replay simulation
[flux2,detp2,vol2,seeds2]=mcxlab(newcfg);

%% data visualization
detpnum=size(newcfg.detpattern,1);
for i=0:(newcfg.srcnum-1)
    for j=0:(detpnum-1)
        jac=flux2.data(:,:,:,i*detpnum+j+1);
        if(~mod(i*detpnum+j,4))
            figure;
        end
        subplot(2,2,mod(i*detpnum+j,4)+1);
        hs=slice(log10(abs(double(jac))),[],[],[1,60]);
        set(hs,'linestyle','none');
        axis equal; colorbar;box on;
        title(['jacobian(wl) of src#',num2str(i+1),' det#',num2str(j+1)]);
%         figure;imagesc(log10(abs(squeeze(jac(15,:,:))))');view([0,0,-1]);title(['at
%         x=15']); %compare vertical crossection
    end
end
