import os
import subprocess

def run_script(script_name):
    print(f"\n{'='*50}\nRunning {script_name}...\n{'='*50}")
    subprocess.run([r".\venv\Scripts\python", script_name], check=True)

def main():
    run_script(r"src\data_collection.py")
    run_script(r"src\preprocessing.py")
    run_script(r"src\eda.py")
    run_script(r"src\models\arima_model.py")
    run_script(r"src\models\prophet_model.py")
    run_script(r"src\models\lstm_model.py")
    run_script(r"src\models\forecast_future.py")
    print("\nAll pipelines executed successfully.")

if __name__ == "__main__":
    main()
