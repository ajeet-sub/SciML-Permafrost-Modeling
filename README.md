# SciML-Permafrost-Modeling
## Description
The following is our code to compute the results for our PINN approach to predict the MAGT and ALT of permafrost. To run the PINN, refer to the PINN_Test.py script, which consists of the training script for the PINN and images of the plots. The GIPL baseline plots are in its own directory as well. While the MAGT and ALT aren't explicitly calculated for the PINN, they can be calculated with the values we get, something that we need to do in future steps. To acquire the results for GIPL, we used a GUI, which we plugged our data into.
### PINN_Test.py directory
- GTNP_Borehole_Dataset/Cleaned/Unformatted Clean Data/PINN_Test.py
- The images of the plots for temperature are also in the Unformatted Clean Data Directory
### GIPL Plots directory
- GTNP_Borehole_Dataset/GIPLResults

## Required Packages
- pytorch
- pandas
- matplotlib
- numpy
