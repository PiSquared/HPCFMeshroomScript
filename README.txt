To use just the photogrammetry program you use python run_alicevison.py 
then the folder where all the generated data should go, 
then the folder with the images, 
then the folder with the alicevision binaries, 
and finally the number of images to scan in. 
So an example would be like "python run_alicevision.py build_files photos Meshroom-2019.1.0/aliceVision/bin 41". 
Then the final model should be in the texturing folder within the generated data folder.

To use the program with the combination of photogrammetry and photometrics you use python run_alicevison_combo.py 
then the folder where all the generated data should go, 
then the folder with the images for the photogrammetrics, 
then the folder with the alicevision binaries, 
then the number of images to scan in,
and finally the folder with the different views for the photometrics.
So an example would be like "python run_alicevision_combo.py build_files photos Meshroom-2019.1.0/aliceVision/bin 41 views". 
Then the final model should be in the texturing folder within the generated data folder, and the bump map should be in the Normal Texture folder.