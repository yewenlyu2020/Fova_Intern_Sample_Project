# The invokeEndPoint function hosted on AWS lambda

import os
import json
import boto3

# grab environment variables
ENDPOINT_NAME = 'pytorch-inference-2020-06-10-13-10-46-759'
runtime = boto3.client('runtime.sagemaker')


def lambda_handler(event, context):
    print("Recieved event: " + json.dumps(event, indent=2))

    payload = event['body']
    print("Payload:" + payload)

    response = runtime.invoke_endpoint(EndpointName=ENDPOINT_NAME,
                                       ContentType='application/json',
                                       Body=payload)

    print(response)
    result = json.loads(response['Body'].read().decode())
    return result
