## MELC DATA ANALYSIS
#### Anna Möller anna.moeller@fau.de

How to use:
1. To segment and analyze the expression of a profile, activate a conda environment using the environment.yml (instructions below).
2. Create direcotires where you want to have the segmentation and expression results and modify the paths in the config.json file (most importantly the datapath).
3. Use /marker_expression/expression_schubert_analysis.ipynb to segment the images and analyze the expression of different profiles.


To use this project, install conda and create an environment from the respective file.

**Step 1: Install Conda**

1. **Download Anaconda**: 
- Go to the Anaconda website at https://www.anaconda.com/products/individual.
- Download the Anaconda Individual Edition for your operating system (Windows, macOS, or Linux).
- Choose the version that matches your system architecture (32-bit or 64-bit).

2. **Install Anaconda**: 
- Once the download is complete, run the installer and follow the installation instructions.
- During installation, you can choose whether to add Anaconda to your system's PATH.
- It's recommended to select this option as it makes it easier to use Conda from the command line.

**Step 2: Create a Conda Environment**

1. **Open a Terminal/Command Prompt**:
   - **Windows**: Press `Win + X`, then select "Windows Terminal" or "Command Prompt."
   - **macOS and Linux**: Use your system's terminal. You can usually find it in the Applications folder (macOS) or by searching for "Terminal" (Linux).

2. **Create a New Conda Environment**: 
- In the terminal, use the following command to create a new Conda environment named "melc_segmentation" and install packages from an environment.yml file:
   
   ```bash
   conda env create -f environment.yml -n melc_segmentation
   ```

   Replace `environment.yml` with the actual path to your environment.yml file if it's not in the current directory.

3. **Activate the Environment**:
- After the environment is created, you need to activate it using the following command:

   ```bash
   conda activate melc_segmentation
   ```

**Step 3: Verify Installation**

1. To verify that your "melc_segmentation" environment is activated, you should see its name in your terminal prompt.

2. You can also check the installed packages by running:

   ```bash
   conda list
   ```

   This will display a list of packages installed in the "melc_segmentation" environment.

That's it! You've successfully installed Conda, created a Conda environment named "melc_segmentation," and installed all the required packages from the environment.yml file. You can now work within this environment for your specific project or tasks. 

