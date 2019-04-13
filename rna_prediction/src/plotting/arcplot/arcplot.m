function color = arcplot(pairings, reliabilities, theta, invx, colorin, plotticks)

    %If reliabilities are not given, just set them all to 1
    if nargin<2
       reliabilities = ones(size(pairings));
    end
    
    %Convert reliabilities to colors
    cmap = colormap(jet(128)); 
    if nargin > 4
        cmap = colorin; 
    end
    
    if nargin < 6
        plotticks = true;
    end
    
    length = max(size(pairings));
    
    
    disp(['Length of sequence: ', num2str(length)]);
    
    %Where the labels should come
    seqticks = [];
    if length < 500;
        seqticks = [(10:10:10*floor(length/10)) length];
    elseif length < 5000;
        seqticks = [(100:100:100*floor((length-50)/100)) length];
    else 
        seqticks = [(500:500:500*floor(length/500)) length];
        
    end
    
    %Plot a circle
    if plotticks
    %t=0:0.1:2*pi;
    t = linspace(0,2*pi,50);
    x = cos(t); y = sin(t); 
    plot(x,y, 'Color', 'k'); hold on;
    end
     axis square;
    set(gca,'YTick',[])
    set(gca,'YColor','w')
    set(gca,'XColor','w')
    set(gca,'XTick',[])
    cb = colorbar('XTickLabel',{'0.00', '0.25', '0.50', '0.75', '1.00'}, 'location', 'southoutside');
    set(cb, 'XTickMode', 'manual');
    cms = max(size(cmap));
    set(cb, 'XTick', [1 floor(cms/4) floor(cms/2) 3*floor(cms/4) cms+1])
    
    
    %Plot seqticks
    if plotticks
    plotseqticks(seqticks, length, theta, invx);
    end
    x1 = -sin(2*pi*transpose((1:1:length))/length);
    y1 = -cos(2*pi*transpose((1:1:length))/length);
    x2 = -sin(2*pi*pairings/length);
    y2 = -cos(2*pi*pairings/length);
    
    for i = 1:length
       % disp(i)
        if reliabilities(i)>=0
           color = cmap(floor(reliabilities(i)*(cms-1))+1,:);
        else 
           %color = 'k';
        end
        
        if pairings(i) > i %pairs only have to be considered once 
            drawarc(i, pairings(i), length, color, theta, invx);
        elseif pairings(i) == 0
            %scatter the point with reliability color 
          %  scatter(x1(i),y1(i), 20, color, 'Marker', 'X');
        else
            %do nothing
        end
    end

end