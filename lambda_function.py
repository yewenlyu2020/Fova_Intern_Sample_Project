# The invokeEndPoint function hosted on AWS lambda

import os
import json
import boto3

# grab environment variables
ENDPOINT_NAME = os.environ['pytorch-inference-2020-06-03-12-13-58-291']
runtime = boto3.client('runtime.sagemaker')


def lambda_handler(event, context):
    print("Recieved event: " + json.dumps(event, indent=2))

    data = json.loads(json.dumps(event))
    payload = data['data']
    print(payload)

    response = runtime.invoke_endpoint(EndpointName=ENDPOINT_NAME,
                                       ContentType='application/json',
                                       Body=payload)

    print(response)
    result = json.loads(response['Body'].read().decode())
    return result
