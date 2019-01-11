function newcfg=mcxdetpattern(cfg)
%
% Format:
%    newcfg=mcxdetpattern(cfg);
%
% Pre-processing for pattern detection, convert rectangular-shaped detector
% to its smallest circumscribed sphere, which is used in mcx. For now up to 
% 1 detector pattern is supported.
%

if(nargin==0)
    error('input field cfg must be defined');
end

if(~isstruct(cfg))
    error('cfg must be a struct or struct array');
end

len=length(cfg);

for i=1:len
    if(isfield(cfg(i),'dettype') && strcmp(cfg(i).dettype,'pattern'))
        if(isfield(cfg(i),'detpos') && isfield(cfg(i),'detparam1') && isfield(cfg(i),'detparam2') && isfield(cfg(i),'detorientation'))
            if((strcmp(cfg(i).detorientation,'+x') || strcmp(cfg(i).detorientation,'-x')) && (cfg(i).detparam1(1,1)~=0 || cfg(i).detparam2(1,1)~=0 ))
                error('the pattern detector is not supported');
            end
            if((strcmp(cfg(i).detorientation,'+y') || strcmp(cfg(i).detorientation,'-y')) && (cfg(i).detparam1(1,2)~=0 || cfg(i).detparam2(1,2)~=0 ))
                error('the pattern detector is not supported');
            end
            if((strcmp(cfg(i).detorientation,'+z') || strcmp(cfg(i).detorientation,'-z')) && (cfg(i).detparam1(1,3)~=0 || cfg(i).detparam2(1,3)~=0 ))
                error('the pattern detector is not supported');
            end
            centroid=cfg(i).detpos(1,1:3)+0.5*cfg(i).detparam1(1,1:3)+0.5*cfg(i).detparam2(1,1:3); %centroid of the detection pattern
            radius=max(sum((centroid-cfg(i).detpos(1,1:3)).^2).^0.5,sum((centroid-(cfg(i).detpos(1,1:3)+cfg(i).detparam1(1,1:3))).^2).^0.5); %use the larger diagonal as the diameter of the sphere detector
            cfg(i).detpos=[centroid,radius];
            newcfg(i)=rmfield(cfg(i),{'dettype','detparam1','detparam2','detorientation'}); %unnecessary for mcx kernel(forward+replay)
        else
            error('input missing for pattern detection');
        end
    end
end

end