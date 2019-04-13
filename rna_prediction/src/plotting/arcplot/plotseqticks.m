function plotseqticks(ticks, length, theta, invx)

    for i = 1:max(size(ticks))
       tick = ticks(i);
       angle = 2*pi*tick/length;
       xpos = 1.15*cos(angle);
       ypos = 1.15*sin(angle);
       
       if invx
            [xpos ypos] = invertx(xpos,ypos);
       end
       [xpos ypos] = rotate(xpos,ypos,theta);
       
       h = text(xpos, ypos, num2str(tick));
       %dims = get(h, 'Extent');
       %set(h, 'Position', [xpos-dims(3)/2, ypos])
       set(h, 'HorizontalAlignment', 'center')
      
       x1 = cos(angle);
       x2 = 1.05*cos(angle);
       y1 = sin(angle);
       y2 = 1.05*sin(angle);
       
       if invx
           [x1 y1] = invertx(x1,y1);
       end
       [x1 y1] = rotate(x1,y1,theta);
       if invx
           [x2 y2] = invertx(x2,y2);
       end
       [x2 y2] = rotate(x2,y2,theta);
       line([x1 x2], [y1 y2], 'Color', 'k')
       
end