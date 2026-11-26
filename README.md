# AI Traffic Congestion Predictor

A beginner-friendly BCA machine-learning project built with Python, Flask,
Pandas and Scikit-learn.

## Features
- Enter hour, weekday, estimated vehicle count, weather and road type.
- Predict Low, Medium or High congestion.
- Show model confidence and a responsive dashboard.
- Train a Random Forest model from a generated sample dataset.

## Important limitation
The included dataset is synthetic and generated for educational demonstration.
The project does not use live traffic sensors, maps, cameras or GPS. Model
confidence is not a guarantee of real-world accuracy. Do not use these
predictions for actual travel or emergency decisions.

## Run in Windows / VS Code

1. Extract the ZIP and open the `AI_Traffic_Congestion_Predictor` folder in VS Code.
2. Open the terminal in that folder.
3. Create a virtual environment:

   ```powershell
   py -m venv venv
   ```

4. Activate it in PowerShell:

   ```powershell
   .\venv\Scripts\Activate.ps1
   ```

   If PowerShell blocks activation, use Command Prompt:

   ```cmd
   venv\\Scripts\\activate.bat
   ```

5. Install packages:

   ```powershell
   py -m pip install -r requirements.txt
   ```

6. Train the model and create the sample CSV:

   ```powershell
   py train_model.py
   ```

7. Start the website:

   ```powershell
   py app.py
   ```

8. Open this address in your browser:

   `http://127.0.0.1:5000`

If `py` is not recognized, install Python 3.10 or newer and select
"Add Python to PATH" during installation.

## Project files
- `app.py` - Flask web application
- `train_model.py` - sample dataset generation and model training
- `traffic_sample.csv` - generated dataset (created after training)
- `traffic_model.pkl` - trained model (created after training)
- `templates/` - HTML pages
- `static/` - CSS and JavaScript

## Model
RandomForestClassifier inside a Scikit-learn Pipeline with one-hot encoding
for categorical inputs. The training script prints a held-out test score.
Because the data is synthetic, that score measures fit to the simulation,
not real-world traffic prediction quality.
