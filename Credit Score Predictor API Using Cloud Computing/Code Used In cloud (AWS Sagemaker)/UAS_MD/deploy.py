import boto3
import tarfile
import json
import sagemaker
from sagemaker.sklearn.model import SKLearnModel

def package_model():

    with tarfile.open("model.tar.gz", "w:gz") as tar:
        tar.add("artifacts/best_model.pkl",arcname="best_model.pkl")


def get_bucket():
    session = sagemaker.Session()
    return session.default_bucket()

REGION = "us-east-1"
INSTANCE_TYPE = "ml.m5.large"
FRAMEWORK_VERSION = "1.4-2"
    
# ---- EDIT THESE ---------------------------------------------------------
BUCKET = get_bucket()
MODEL_S3_KEY = "model/model.tar.gz"
ENDPOINT_NAME = "CreditScore-endpoint"
# -------------------------------------------------------------------------

def upload_model():

    s3 = boto3.client("s3")
    s3.upload_file("model.tar.gz",BUCKET, MODEL_S3_KEY)


def get_lab_role_arn() -> str:
    iam = boto3.client("iam")
    return iam.get_role(RoleName="LabRole")["Role"]["Arn"]


def main() -> None:
    boto3.setup_default_session(region_name=REGION)
    sm_session = sagemaker.Session()
    role_arn = get_lab_role_arn()
    model_s3_uri = f"s3://{BUCKET}/{MODEL_S3_KEY}"

    print(f"Role:      {role_arn}")
    print(f"Model URI: {model_s3_uri}")
    print(f"Endpoint:  {ENDPOINT_NAME}")

    model = SKLearnModel(
        model_data=model_s3_uri,
        role=role_arn,
        entry_point="inference.py",
        source_dir=".",
        framework_version=FRAMEWORK_VERSION,
        sagemaker_session=sm_session,
    )

    print("\nDeploying endpoint (5-8 minutes)...")
    predictor = model.deploy(
        initial_instance_count=1,
        instance_type=INSTANCE_TYPE,
        endpoint_name=ENDPOINT_NAME
    )

    print("Testing endpoint with sample data...")
    sample = {
    "instances": [
        {
            "Month": "January","Age": 40,
            "Annual_Income": 50000,
            "Num_Bank_Accounts": 2,
            "Num_Credit_Card": 3,
            "Interest_Rate": 1.5,
            "Delay_from_due_date": 2,
            "Num_Credit_Inquiries": 1,
            "Credit_Utilization_Ratio": 25.5,
            "Num_of_Loan": 1,
            "Num_of_Delayed_Payment": 0,
            "Changed_Credit_Limit": 2.5,
            "Outstanding_Debt": 1000,
            "Amount_invested_monthly": 500,
            "Monthly_Balance": 2500,
            "Monthly_Inhand_Salary": 4500,
            "Total_EMI_per_month": 150,
            "Credit_History_Age_Months": 60,
            "Occupation": "Engineer",
            "Type_of_Loan": "Personal Loan",
            "Credit_Mix": "Good",
            "Payment_of_Min_Amount": "Yes",
            "Payment_Behaviour": "Low_spent_Small_value_payments"}]}

    runtime = boto3.client("sagemaker-runtime", region_name=REGION)
    response = runtime.invoke_endpoint(
        EndpointName=ENDPOINT_NAME,
        ContentType="application/json",
        Accept="application/json",
        Body=json.dumps(sample)
    )
    print("\nSmoke test response:")
    print(response["Body"].read().decode("utf-8"))

if __name__ == "__main__":

    package_model()
    upload_model()
    main()
