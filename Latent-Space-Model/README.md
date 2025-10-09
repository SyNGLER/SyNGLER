## This file contains code for our latent space model.

### Project Structure
- **config/**
  - `default.json`: Default configuration file for model parameters.
- **generate.ipynb**: Interactive Jupyter Notebook for demonstrating and debugging data generation and model execution.
- **simulated_data/**
  - `generator.py`: Script for generating simulated network data.
  - `LSM_source.py`: Core implementation of the Latent Space Model.
  - `run.py`: Main entry script to run the Latent Space Model.

### Usage
1. **Data Generation**  
   Use `simulated_data/generator.py` to generate simulated network data.  
   ```bash
   python simulated_data/generator.py
   ```

2. **Run Latent Space Model**
   Use `simulated_data/run.py` to run the LSM with specified configurations.

   ```bash
   python simulated_data/run.py --config config/default.json
   ```

3. **Interactive Experiments**
   For step-by-step exploration, you can run `generate.ipynb` in Jupyter Notebook to generate data and test the model interactively.

### Requirements

* Python >= 3.8
* Main dependencies:

  * numpy
  * os
  * pathlib
  * csv
  * json
  * pickle
  * sys
  * absl-py
  * tqdm

### Notes

* Modify experimental and model parameters in `config/default.json`.
* Results will be saved in the working directory for further analysis.

