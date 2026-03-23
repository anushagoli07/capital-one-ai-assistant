import boto3
from botocore.exceptions import ClientError
from core.config import settings
import json
import os

class S3Storage:
    def __init__(self):
        # Connect to AWS S3
        self.s3 = boto3.client(
            "s3",
            region_name=settings.aws_region
        )
        self.bucket = settings.aws_bucket_name
        print(f"S3 Storage connected to: {self.bucket}")

    def upload_file(self, local_path, s3_key):
        try:
            self.s3.upload_file(
                local_path,
                self.bucket,
                s3_key
            )
            print(f"Uploaded: {local_path} → s3://{self.bucket}/{s3_key}")
            return True
        except ClientError as e:
            print(f"Upload failed: {e}")
            return False

    def upload_json(self, data, s3_key):
        try:
            self.s3.put_object(
                Bucket=self.bucket,
                Key=s3_key,
                Body=json.dumps(data),
                ContentType="application/json"
            )
            print(f"Uploaded JSON → s3://{self.bucket}/{s3_key}")
            return True
        except ClientError as e:
            print(f"Upload failed: {e}")
            return False

    def download_file(self, s3_key, local_path):
        try:
            self.s3.download_file(
                self.bucket,
                s3_key,
                local_path
            )
            print(f"Downloaded: s3://{self.bucket}/{s3_key} → {local_path}")
            return True
        except ClientError as e:
            print(f"Download failed: {e}")
            return False

    def list_files(self, prefix=""):
        try:
            response = self.s3.list_objects_v2(
                Bucket=self.bucket,
                Prefix=prefix
            )
            files = []
            if "Contents" in response:
                for obj in response["Contents"]:
                    files.append(obj["Key"])
            return files
        except ClientError as e:
            print(f"List failed: {e}")
            return []

if __name__ == "__main__":
    storage = S3Storage()

    # Upload financial products to S3
    success = storage.upload_file(
        "data/financial_products.json",
        "data/financial_products.json"
    )

    if success:
        print("Financial products uploaded to S3!")
        files = storage.list_files("data/")
        print(f"Files in S3: {files}")
