% SupportEstimation.m: This script takes in the array of cross-sectional 
% areas stored in an Excel file and calculate an approximation of 
% the support volume required for a specific print job.
% 
% Author: Steve Tran                           
% Date created: 27/7/2019

% Clear existing variables.
clc;
clear;
% Input the array of cross-sectional areas stored in an Excel file
% Since the algorithm works from the top down while the array is stored
% from the bottom up, the array has to be flipped.
csArea = fliplr(transpose(xlsread('spinner_standingup.xlsx',1,'B2:B100000')));
% Initialise variables such to use for calculation.
sliceHeight = 0.1;
supportApprox = 0;
areaRef = csArea(1);            % Storing the first area as reference.
radiusRef = sqrt(areaRef/pi);   % Calculate the radius of the reference area circle.

% Loop through the array of cross-sectional area to calculate total 
% support volume.
for i=2:length(csArea)
    % Storing the area of the next slice
    areaBelow = csArea(i);      
    % Calculate the threshold area.
    areaThreshold = pi*(radiusRef - sliceHeight)^2;     
    % Comparing the threshold area with the area of the next slice.
    if (areaThreshold <= areaBelow)     
        % Reference area and radius are updated if no support is required.
        areaRef = areaBelow;   
        radiusRef = sqrt(areaRef/pi);   
    else
        % Sum up the support volume required for each slice to obtain
        % the total support volume.
        supportApprox = supportApprox + (areaRef - areaBelow)*sliceHeight;  
    end
end

% Print out the total support volume for the print job
fprintf('The approximation of the support volume for this print job is %.2f mm^3 \n', supportApprox)