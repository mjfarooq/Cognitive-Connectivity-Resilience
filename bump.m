function y = bump(z,h)
    
    if z >= 0 && z < h
        y = 1;
    elseif z >= h && z <= 1
        y = 0.5*(1 + cos(pi*(z-h)/(1-h)));
    elseif z < 0
        y = 1;
    else 
        y = 0;
    end
end
