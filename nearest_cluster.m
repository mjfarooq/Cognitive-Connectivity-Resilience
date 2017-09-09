function id= nearest_cluster(x,C) 
   Mat = C - repmat(x, size(C,1),1);
   dist = sqrt(sum(Mat.^2, 2));
   [~, id] = min(dist);
end
