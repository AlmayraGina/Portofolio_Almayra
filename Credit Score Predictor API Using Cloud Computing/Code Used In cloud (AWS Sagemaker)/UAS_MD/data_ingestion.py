from pathlib import Path
import pandas as pd
import os

class DataIngestion:

    def __init__(self, filename="data_D_(2).csv"):

        self.filename = filename

        # SageMaker Processing Job
        container_input = Path("/opt/ml/processing/input")
        container_output = Path("/opt/ml/processing/ingested")

        # Local testing
        local_input = Path("/home/ec2-user/SageMaker/UAS_MD")
        local_output = Path("/home/ec2-user/SageMaker/UAS_MD/ingested")

        if container_input.exists():
            self.input_dir = container_input
            self.output_dir = container_output
            print("Running inside SageMaker container")
        else:
            self.input_dir = local_input
            self.output_dir = local_output
            print("Running locally")

        self.input_file = self.input_dir / self.filename
        self.output_file = self.output_dir / self.filename

    def run(self):

        print("\n--- Step 1: Data Ingestion ---")

        self.output_dir.mkdir(parents=True, exist_ok=True)

        print(f"Looking for: {self.input_file}")

        if not self.input_file.exists():
            print(f"❌ File not found: {self.input_file}")
            print(f"Files available: {list(self.input_dir.iterdir())}")
            raise FileNotFoundError(self.input_file)

        df = pd.read_csv(self.input_file)

        if df.empty:
            raise ValueError("Dataset is empty")

        df.to_csv(self.output_file, index=False)

        print(f"✅ Data ingested successfully")
        print(f"Saved to: {self.output_file}")

        return self.output_file


if __name__ == "__main__":
    ingestor = DataIngestion()
    ingestor.run()