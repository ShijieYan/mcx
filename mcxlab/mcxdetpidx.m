function [replaydetidx,newdata,newseeds]=mcxdetpidx(detpos,detparam1,detparam2,detorientation,detp,seeds,unitinmm)
%
% Format:
%    [replaydetidx,newdetp,newseeds]=mcxdetpidx(detpos,detparam1,detparam2,detorientation,detp,seeds,unitinmm)
%
% Post-processing for pattern detection. For now only rectangular-shaped
% detection is supported.
%

if(~isfield(detp,'p'))
    error('photon exiting positions are missing');
end

if(~ischar(detorientation))
    error('Detector orientation is not defined');
end

detpos=detpos*unitinmm; 
detparam1=detparam1*unitinmm;
detparam2=detparam2*unitinmm;
orientation=0;
detpTo=zeros(size(detp.p,1),1);
detpTo2=zeros(size(detp.p,1),2);

if(strcmp(detorientation(1),'+'))
    orientation=1;
else
    orientation=-1;
end
    
if(strcmp(detorientation(2),'x'))
    detcorner=detpos(1,2:3);
    range=detparam1(1,2:3)+detparam2(1,2:3); 
    detFrom=detpos(1,1);
    detpTo=detp.p(:,1);
    detpTo2=detp.p(:,2:3);
elseif(strcmp(detorientation(2),'y'))
    detcorner=detpos(1,[1 3]);
    range=detparam1(1,[1 3])+detparam2(1,[1 3]);
    detFrom=detpos(1,2);
    detpTo=detp.p(:,2);
    detpTo2=detp.p(:,[1 3]);
elseif(strcmp(detorientation(2),'z'))
    detcorner=detpos(1,1:2);   %equivalent to x0,y0 in mmcdetidx.m
    range=detparam1(1,1:2)+detparam2(1,1:2); %equivalent to xrange and yrange in mmcdetidx.m
    detFrom=detpos(1,3);
    detpTo=detp.p(:,3);
    detpTo2=detp.p(:,1:2);
else
    orientation=0;
end
  
if(~orientation)
    error('Detector orientation is not correctly defined or current orientation is not supported yet');
end

%% next, sort out the photon belongs to the pattern detector area
maxvoidstep=1000; %an important internal parameter in mcx, needed in post processing
detFrom=detFrom+maxvoidstep;
detTo=detFrom+orientation*eps(single(detFrom))-maxvoidstep;

logic=(detpTo(:,1)==detTo); %sort out the photon belongs to the pattern detector plan(x,y or z orientated)
newdata=detp.data(:,logic);
    newseeds=seeds.data(:,logic);
detpTo2=detpTo2(logic,:);

patternsize = [detparam1(1,4),detparam2(1,4)]; %equivalent to xsize and ysize in mmcdetidx.m

index1 = floor((detpTo2(:,1)-detcorner(1,1)) / range(1,1) * patternsize(1));
index2 = floor((detpTo2(:,2)-detcorner(1,2)) / range(1,2) * patternsize(2));

logic = index1>=0 & index1<patternsize(1) & index2>=0 & index2<patternsize(2);
newdata=newdata(:,logic); %sort out the photon belongs to the pattern detector rectangular area
    newindex1 = index1(logic);
    newindex2 = index2(logic);
    newseeds=newseeds(:,logic);

replaydetidx = uint32(newindex2 * patternsize(2) + newindex1);%hierachy always: x<y<z

end
