
import boto3
import io
import os
import requests
import json
import pandas as pd

BUCKET = 'datalake'
KEY = 'raman-ml-service/monitordata.parquet'

def write_records(inputs,output):
    s3 = boto3.client('s3')
    d = {}
    for i in inputs:
        d[i['name']] = i['data']
    d['predict']=output
    ndf = pd.DataFrame(d)
    try:
        obj = s3.get_object(Bucket=BUCKET, Key=KEY)
        df = pd.read_parquet(io.BytesIO(obj['Body'].read()))
    except Exception as e:
        df = pd.DataFrame()

    df = pd.concat([df,ndf])
    df = df.tail(30000)

    df.to_parquet("monitoringdata.parquet")
    s3.upload_file("monitoringdata.parquet", BUCKET, KEY)

def init(context):
    url = os.getenv("SERVICE_URL")
    if not url:
        raise Exception("Missing SERVICE_URL env variable")

    setattr(context, "service", url)

def serve(context, event):
    context.logger.info(f"Received event: {event}")

    if isinstance(event.body, bytes):
        body = json.loads(event.body)
    else:
        body = event.body

    inputs = body["inputs"]
    res = requests.post(f"http://{context.service}", json=body)
    output_json = json.loads(res.text)    

    write_records(inputs, output_json['outputs'][0]['data'][0])

    return output_json
