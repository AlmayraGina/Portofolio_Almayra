from pathlib import Path
from data_ingestion import DataIngestion
from train import CreditScoreTrainer
from evaluation import ModelEvaluator

class CreditScorePipeline:

    def __init__(self, raw_data_path, recall_threshold):

        self.raw_data_path = Path(raw_data_path)
        self.ingestor = DataIngestion(
            self.raw_data_path)
        self.trainer = CreditScoreTrainer()
        self.evaluator = ModelEvaluator()
        self.recall_threshold = recall_threshold

    def execute(self):
        ingested_file_path = self.ingestor.run()
        run_id, x_test, y_test = (self.trainer.run(ingested_file_path))
        accuracy, precision, recall = (
            self.evaluator.run(run_id,x_test,y_test))

        if recall >= self.recall_threshold:
            print("Approved")
        else:
            print("Rejected")

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall}

if __name__ == "__main__":

    DATA_INPUT = (Path(__file__).parent /"data_D_(2).csv")
    pipeline = CreditScorePipeline(raw_data_path=DATA_INPUT,recall_threshold=0.8)
    results = pipeline.execute()
    print(results)