function [xn yn] = rotate(x,y,thetadeg)
%ROTATE Rotate vector pair x,y by angle theta. 

%Convert degrees to radians
theta = thetadeg*pi/180;
    
xn = cos(theta)*x - sin(theta)*y;
yn = sin(theta)*x + cos(theta)*y;

end

