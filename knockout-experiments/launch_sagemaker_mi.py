#!/usr/bin/env python3
"""
Launch the mech interp experiment on SageMaker Processing.

Usage:
  pip install sagemaker boto3
  python launch_sagemaker_mi.py

Prerequisites:
  - AWS credentials configured (aws configure)
  - SageMaker execution role ARN
  - S3 bucket for output
"""

import sagemaker
from sagemaker.processing import ScriptProcessor

# CONFIGURE THESE:
ROLE = "arn:aws:iam::YOUR_ACCOUNT:role/YOUR_SAGEMAKER_ROLE"  # Your SageMaker role
BUCKET = "YOUR_BUCKET"  # S3 bucket for output
REGION = "eu-central-1"

session = sagemaker.Session(default_bucket=BUCKET)

processor = ScriptProcessor(
    image_uri=f"763104351884.dkr.ecr.{REGION}.amazonaws.com/pytorch-training:2.1.0-gpu-py310-cu121-ubuntu20.04-sagemaker",
    role=ROLE,
    instance_count=1,
    instance_type="ml.g4dn.xlarge",
    command=["python3"],
    sagemaker_session=session,
    max_runtime_in_seconds=7200,  # 2 hour max
)

processor.run(
    code="mech_interp_rashomon_gpu.py",
    outputs=[
        sagemaker.processing.ProcessingOutput(
            output_name="results",
            source="/opt/ml/processing/output",
            destination=f"s3://{BUCKET}/mech-interp-results/",
        )
    ],
    arguments=[],
)

print(f"Job submitted. Check SageMaker console for status.")
print(f"Results will be in s3://{BUCKET}/mech-interp-results/")
