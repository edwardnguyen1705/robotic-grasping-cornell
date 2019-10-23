%% script to test dataPreprocessing
%% created by Fu-Jen Chu on 09/15/2016

close all
clear
clc
%%
%parpool(4)
%addpath('E:/Robotics_Lab/Robotics/Grasp/code/robot-grasp-detection/datasets/cornell/')

% generate list for splits
list = [100:949 1000:1034];
% create a list of random number from 1 to length(list)
list_idx = randperm(length(list));  % length(list) = 885
% 4/5 * 885 = 708 for training, only indices
train_list_idx = list_idx(length(list)/5+1:end);
% 1/5 * 885 = 177 for testing, only indices
test_list_idx = list_idx(1:length(list)/5);
% get actual image ids for training and testing
train_list = list(train_list_idx);
test_list = list(test_list_idx);

imgTestOutDir = 'E:/Robotics_Lab/Robotics/Grasp/code/grasp_multiObject/grasp_multiObject_multiGrasp/dataset/grasp/Images_Test/Cropped320_rgd';
imgDataOutDir = 'E:/Robotics_Lab/Robotics/Grasp/code/grasp_multiObject/grasp_multiObject_multiGrasp/dataset/grasp/Images';
annotationDataOutDir = 'E:/Robotics_Lab/Robotics/Grasp/code/grasp_multiObject/grasp_multiObject_multiGrasp/dataset/grasp/Annotations';
imgSetTrain = 'E:/Robotics_Lab/Robotics/Grasp/code/grasp_multiObject/grasp_multiObject_multiGrasp/dataset/grasp/ImageSets/train.txt'; 
imgSetTest = 'E:/Robotics_Lab/Robotics/Grasp/code/grasp_multiObject/grasp_multiObject_multiGrasp/dataset/grasp/ImageSets/test.txt';
%%
for folder = 1:10
    display(['processing folder ' int2str(folder)])
    % sprintf('%02d',folder): int to str, %02d: 01, 02, etc.
    % raw data from Cornell dataset
    imgDataDir = ['E:/Robotics_Lab/Robotics/Grasp/code/robot-grasp-detection/datasets/cornell/' sprintf('%02d',folder)];
    txtDataDir = ['E:/Robotics_Lab/Robotics/Grasp/code/robot-grasp-detection/datasets/cornell/' sprintf('%02d',folder)];   
    % get a list of png files in the imgDataDir folder
    imgFiles = dir([imgDataDir '/*.png']);
    % get a list of txt files in the txtDataDir folder
    txtFiles = dir([txtDataDir '/*pos.txt']);

    logfileID = fopen('log.txt','a');
    %mainfileID = fopen(['/home/fujenchu/projects/deepLearning/deepGraspExtensiveOffline/data/grasps/scripts/trainttt' sprintf('%02d',folder) '.txt'],'a');
    % process each folder, cornell/01,cornell/02, etc.
    for idx = 1:length(imgFiles) 
        %% display progress
        tic
        display(['processing folder: ' sprintf('%02d',folder) ', imgFiles: ' int2str(idx)])

        %% reading data
        imgName = imgFiles(idx).name;
        % returns the path name, file name, and extension for the specified file.
        [pathstr,imgname,ext] = fileparts(imgName);
        img = imread([imgDataDir '/' imgname '.png']);
        % imgname(4:7): only get number part, 0100, 0101, etc.
        filenum = str2num(imgname(4:7));
        if(any(test_list == filenum))
            % Open or create new file for writing. Append data to the end of the file.
            file_writeID = fopen(imgSetTest,'a');
            % write ful paths of testing images to a txt file 
            fprintf(file_writeID, '%s\n', [imgTestOutDir '/' imgname '_preprocessed_1.png' ] );
            fclose(file_writeID);
            
            imwrite(img, [imgTestOutDir '/' imgname '_preprocessed_1.png']);
            
            % write annotation txt files for testing?
            
            continue; % if test_list == filenum, then skip the blow code
        end

        txtName = txtFiles(idx).name;
        % returns the path name, file name, and extension for the specified file.
        [pathstr,txtname, ext] = fileparts(txtName);

        
        fileID = fopen([txtDataDir '/' txtname '.txt'],'r');
        sizeA = [2 100]; % 2: x and y, 100/4 = 25 -> max 25 rectangles
        % read ground truth rectangles from *pos.txt
        bbsIn_all = fscanf(fileID, '%f %f', sizeA);
        fclose(fileID);

        %% data pre-processing
        % [imagesOut, bbsOut] = dataPreprocessing_fasterrcnn(img, bbsIn_all, 224, 5, 5);
        [imagesOut, bbsOut] = dataPreprocessing_fasterrcnn(img, bbsIn_all, 224, 2, 2);

        % for each augmented image
        for i = 1:1:size(imagesOut,2) % length(imagesOut)
            % for each bbs
            % write annotation txt file
            file_writeID = fopen([annotationDataOutDir '/' imgname '_preprocessed_' int2str(i) '.txt'],'w');
            printCount = 0;
            for ibbs = 1:1:size(bbsOut{i},2) % length(bbsOut{i}) = 4
              A = bbsOut{i}{ibbs}; % a rectangle (4 points (x,y))
              xy_ctr = sum(A,2)/4; x_ctr = xy_ctr(1); y_ctr = xy_ctr(2);
              width = sqrt(sum((A(:,1) - A(:,2)).^2)); height = sqrt(sum((A(:,2) - A(:,3)).^2));
              % a rect: 1-2-3-4-1: 1-2 and 3-4 are long egdes, 3-4 and 4-1 are short edges
              % theta is the angle between the 1-2 (or 2-1) and the x axis
              % For real values of X, atan(X) returns values in the interval [-?/2, ?/2]
              % if x1 > x2
              if(A(1,1) > A(1,2))
                  % (y2 - y1) / (x1 - x2)
                  theta = atan((A(2,2)-A(2,1))/(A(1,1)-A(1,2)));
              else % x1 <= x2
                  %(y1 - y2) / (x2 - x1)
                  theta = atan((A(2,1)-A(2,2))/(A(1,2)-A(1,1))); % note y is facing down
              end  

              % process to fasterrcnn
              x_min = x_ctr - width/2; x_max = x_ctr + width/2;
              y_min = y_ctr - height/2; y_max = y_ctr + height/2;
              %if(x_min < 0 || y_min < 0 || x_max > 224 || y_max > 224) display('yoooooooo'); end
              % if the one point of the rect is outside the image coordinates, then the .txt is empty
              if((x_min < 0 && x_max < 0) || (y_min > 224 && y_max > 224) || (x_min > 224 && x_max > 224) || (y_min < 0 && y_max < 0)) 
                  disp('xxxxxxxxx'); 
                  break; 
              end
              
              % theta is in [-pi/2, pi/2] <-> class 1, ..., class 19
              % class 0: no possible orientation
              % theta+90: the angle between the short edge and the axis 
              cls = round((theta/pi*180+90)/10) + 1;
              % write as lefttop rightdown, Xmin Ymin Xmax Ymax, ex: 261 109 511 705  (x水平 y垂直)
              fprintf(file_writeID, '%d %f %f %f %f\n', cls, x_min, y_min, x_max, y_max );   
              printCount = printCount+1;
            end
            
            fclose(file_writeID);
            
            if(printCount == 0) 
                fprintf(logfileID, '%s\n', [imgname '_preprocessed_' int2str(i) ]);
                
            end

            
            imwrite(imagesOut{i}, [imgDataOutDir '/' imgname '_preprocessed_' int2str(i) '.png']); 
            % if is better to do: if(printCount) 
            % write filename to imageSet 
            file_writeID = fopen(imgSetTrain,'a');
            fprintf(file_writeID, '%s\n', [imgname '_preprocessed_' int2str(i) ] );
            fclose(file_writeID);

        end
        toc
    end
    %fclose(mainfileID);
end
