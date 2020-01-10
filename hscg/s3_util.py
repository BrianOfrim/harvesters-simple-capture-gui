import os
import pathlib
from typing import List

import boto3
import botocore


def s3_bucket_exists(name: str) -> bool:
    s3 = boto3.client("s3")
    try:
        s3.head_bucket(Bucket=name)
    except botocore.exceptions.ClientError as e:
        print(e)
        return False
    return True


def file_exists(bucket_name: str, s3_object_path: str) -> None:
    s3 = boto3.resource("s3")
    try:
        s3.Object(bucket_name, s3_object_path).load()  # pylint: disable=no-member
    except botocore.exceptions.ClientError as e:
        if e.response["Error"]["Code"] == "404":
            return False
        else:
            raise
    else:
        return True


def upload_files(
    bucket_name, files_to_send: List[str], s3_destination_object_dir: str
) -> None:
    s3 = boto3.client("s3")
    for file_index, file_to_send in enumerate(files_to_send):
        s3_destination_object_path = os.path.join(
            s3_destination_object_dir, os.path.basename(file_to_send)
        )
        try:
            if file_exists(bucket_name, s3_destination_object_path):
                print(
                    "S3 object already exists %s:%s, %i/%i"
                    % (
                        bucket_name,
                        s3_destination_object_dir,
                        file_index + 1,
                        len(files_to_send),
                    )
                )
                continue
            s3.upload_file(file_to_send, bucket_name, s3_destination_object_path)
        except botocore.exceptions.ClientError as e:
            print(e)
            continue
        print(
            "Uploading file to %s:%s, %i/%i"
            % (
                bucket_name,
                s3_destination_object_path,
                file_index + 1,
                len(files_to_send),
            )
        )
