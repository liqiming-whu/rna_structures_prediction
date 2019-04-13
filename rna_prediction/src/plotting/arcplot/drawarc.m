function drawarc (p1, p2, seqlength, color, theta, invx)

%Plots an arc on a unit circle between p1 and p2, where seqlength is the
%circumference scaling factor

%t = 0:0.01:2*pi;
%plot(sin(t),cos(t)); hold on; axis square;

phi1 = 2*pi*p1/seqlength;
phi2 = 2*pi*p2/seqlength;
x1 = cos(phi1);
x2 = cos(phi2);
y2 = sin(phi2);
y1 = sin(phi1);

midpoint = [(x1+x2) (y1+y2)];
u = midpoint / sqrt(midpoint(1)^2 + midpoint(2)^2);

dphi = phi2-phi1;
r2 = tan(dphi/2);
d =sqrt(1 + r2^2);
centrum = d*u;

lowboundangle = phi1 + 3* pi/2;
highboundangle = phi2 + pi/2;

%linear spacing
%tn = min(lowboundangle,highboundangle):0.05:max(lowboundangle, highboundangle);
tn = linspace(min(lowboundangle,highboundangle),max(lowboundangle, highboundangle),30);


x = centrum(1) + r2*cos(tn);
y = centrum(2) + r2*sin(tn);
if invx
   [x y] = invertx(x,y);
end
[x y] = rotate(x,y,theta);

plot(x,y, 'Color', color)