function [imagesOut, bbsOut] = dataPreprocessing_fasterrcnn( imageIn, bbsIn_all, cropSize, translationShiftNumber, roatateAngleNumber)
% dataPreprocessing function perfroms
% 1) croping
% 2) padding
% 3) rotatation
% 4) shifting
%
% for an input image with a bbs as 4 points, 
% dataPreprocessing_fasterrcnn outputs a set of images with corresponding bbs.
%
%
% Inputs: 
%   imageIn: input image (480 by 640 by 3)
%   bbsIn: bounding box (2 by 4)
%   cropSize: output image size
%   shift: shifting offset
%   rotate: rotation angle
%
% Outputs:
%   imagesOut: output images (n images)
%   bbsOut: output bbs according to shift and rotation
%
% [imagesOut, bbsOut] = dataPreprocessing_fasterrcnn(img, bbsIn_all, 224, 5, 5);
%% created by Fu-Jen Chu on 09/15/2016

debug_dev = 0;
debug = 0;
%% show image and bbs
if(debug_dev)
    figure(1); imshow(imageIn); hold on;
    x = bbsIn_all(1, [1:3]);
    y = bbsIn_all(2, [1:3]);
    plot(x,y); hold off;
end

%% crop image and padding image
% original image size: 640x480x3
% center cropped 351x351
% [145 65 351 351] = [xmin ymin width height]
imgCrop = imcrop(imageIn, [145 65 351 351]); 

% padding to 501 by 501
imgPadding = padarray(imgCrop, [75 75], 'replicate', 'both');

count = 1;
% 5*5^2 = 125, 5^2: for x and y
for i_rotate = 1:roatateAngleNumber*translationShiftNumber*translationShiftNumber
    % random roatateAngle. 
    % randi(360) returns a pseudorandom scalar integer between 1 and 360
    theta = randi(360)-1;
    %theta = 0;

    % random translationShift. dx is in [1, 50]
    dx = randi(101)-51;
    %dx = 0;

    %% rotation and shifting
    % random translationShift. dy is in [1, 50]
    dy = randi(101)-51;
    %dy = 0;
    
    % rotate an image around its center by an angle in CCW dir, using nearest neighbor interpolation
    imgRotate = imrotate(imgPadding, theta);
    
    if(debug_dev)figure(2); imshow(imgRotate);end
    
    % size(imgRotate,1) depends on theta,imgRotate is a squared image
    % center cropped 321x321. 
    imgCropRotate = imcrop(imgRotate, [size(imgRotate,1)/2-160-dx size(imgRotate,1)/2-160-dy 320 320]);
    
    if(debug_dev)figure(3); imshow(imgCropRotate);end 
    % resize from 321x321 to 224x224
    imgResize = imresize(imgCropRotate, [cropSize cropSize]); % cropSize = 224
    
    if(debug)figure(4); imshow(imgResize); hold on;end

    %% modify bbs
    % m = 2, for x and y coordinates. n is num of points (rows in a pos.txt)
    [m, n] = size(bbsIn_all);
    bbsNum = n/4;   % num of max rects (100/4)
    
    countbbs = 1;
    for idx = 1:bbsNum
      % idx*4-3:idx*4: 1:4, 5:8, etc
      bbsIn =  bbsIn_all(:,idx*4-3:idx*4); % size(bbsIn) = [2, 4]
      if(sum(sum(isnan(bbsIn)))) continue; end
      % repmat([320; 240], 1, 4): [320, 320, 320, 320; 
      %                            240, 240, 240, 240]
      % size(repmat([320; 240], 1, 4)) = [2, 4]
      % (320, 240): center of original image
      % shift the left corner coordinates to the center coordinates
      bbsInShift = bbsIn - repmat([320; 240], 1, 4);
      % rotate points in the xy-plane CW (because y directed down) through an angle θ about the origin z axis
      R = [cos(theta/180*pi) -sin(theta/180*pi); sin(theta/180*pi) cos(theta/180*pi)];
      % R' = R(-θ): CCW
      bbsRotated = (bbsInShift'*R)'; % = R' * bbsInShift
      % bbsRotated + repmat([160; 160], 1, 4) + repmat([dx; dy], 1, 4): shift to the left corner coordinates of the 321x321 image (imgCropRotate) 
      % shift back to the left corner coordinates of the 224x224 image (imgResize)
      bbsInShiftBack = (bbsRotated + repmat([160; 160], 1, 4) + repmat([dx; dy], 1, 4))*cropSize/320;
      
      if(debug)
        figure(4)
        x = bbsInShiftBack(1, [1:4 1]);
        y = bbsInShiftBack(2, [1:4 1]);
        plot(x,y); hold on; pause(0.01);
      end
      
      % count: image idn, countbbs: box idn in the image
      bbsOut{count}{countbbs} = bbsInShiftBack;
      countbbs = countbbs + 1;
    end
    
    imagesOut{count} = imgResize;
    count = count +1;

   
end



end
