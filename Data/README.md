This README file includes the description of the files located in Data
as well as explanation of different versions where applicable.

This folder contains the data collected and used in my project.

=================================

FILES:
Final Data.xlsx - The spreadsheet containing the raw dataset. Each row is a data points and each column is a feature/output.  
The spreadsheet consists of three sheet: Training Data, Testing Data, Other Parameters for Reference.  
Training Data: Raw training data.  
Column A (blue) contains the file names, orientations and photos showing the orientation. The naming convention is the same for model files and orientation photos.  
Columns B-J (yellow) contain the values for printing features.  
Columns K-P (orange) contain the values for model features.  
Columns Q-AA (white) contain the time specification and days, hours, minutes of printing time.  
Columns AB-AD (red) contain the values for printing time and material usage to be used as output.  
Testing Data: Raw testing data.  
Follows the same convention as Training Data.  
Other Parameters for Reference: Containing other parameters in Ultimaker Cura 4.0 that were not investigated.  
These were recorded to ensure consistency for all datapoints.

Final Data - Processed.xlsx - The spreadsheet containing the processed dataset that can be used directly in feature selection and machine learning code.  
Does not contain name and orientation of the model. Each row is a data points and each column is a feature/output. Each data point is processed.  
The spreadsheet consists of four sheet: Training Randomised, Training Scaled, Testing Randomised, Testing Scaled.  
Training Randomised: Training datapoints do not change value but are randomised (randomised rows)  
Columns A-I (yellow) contain the values for printing features.  
Columns J-O (orange) contain the values for model features.  
Columns P-R (red) contain the values for printing time and material usage to be used as output.  
Training Scaled: Training datapoints are randomised and min-max normalised for each feature (randomised rows, normalised columns)  
Follows the same convention as Training Randomised.  
Testing Randomised: Testing datapoints do not change value but are randomised (randomised rows)  
Follows the same convention as Training Randomised.  
Testing Scaled: Testing datapoints are randomised and min-max normalised for each feature (randomised rows, normalised columns)  
Follows the same convention as Training Scaled.
