import boto3
import botocore
from botocore.exceptions import ClientError
import config
#dynamodb = boto3.resource('dynamodb')
#table = dynamodb.Table('data2')
s3 = boto3.resource('s3')

def delete_dax_table(table_name=None):
    """
    Deletes the demonstration table.

    :param dyn_resource: Either a Boto3 or DAX resource.
    """
    #if dyn_resource is None:
    dyn_resource = boto3.resource('dynamodb')

    table = dyn_resource.Table(table_name)
    table.delete()

    print(f"Deleting {table.name}...")
    table.wait_until_not_exists()


def create_dax_table(params):
    """
    Creates a DynamoDB table.

    :param dyn_resource: Either a Boto3 or DAX resource.
    :return: The newly created table.
    """
    dyn_resource = boto3.resource('dynamodb')

    table = dyn_resource.create_table(**params)
    print(f"Creating tabla...")
    table.wait_until_exists()
    print(f"Created tabla {table.name}")
    return table

def creandoTablaIngesta(tablaName):
    params = {
        'TableName': tablaName,
        'KeySchema': [
            {'AttributeName': 'idFile', 'KeyType': 'HASH'},
            {'AttributeName': 'uuid4', 'KeyType': 'RANGE'}
        ],
        'AttributeDefinitions': [
            {'AttributeName': 'idFile', 'AttributeType': 'S'},
            {'AttributeName': 'uuid4', 'AttributeType': 'S'}
        ],
        'ProvisionedThroughput': {
            'ReadCapacityUnits': 10,
            'WriteCapacityUnits': 10
        }
    }
    dax_table = create_dax_table(params)

def creandoTablaModelos(tablaName):
    params = {
        'TableName': tablaName,
        'KeySchema': [
            {'AttributeName': 'idFile', 'KeyType': 'HASH'}
        ],
        'AttributeDefinitions': [
            {'AttributeName': 'idFile', 'AttributeType': 'S'}
        ],
        'ProvisionedThroughput': {
            'ReadCapacityUnits': 10,
            'WriteCapacityUnits': 10
        }
    }
    dax_table = create_dax_table(params)

if __name__ == '__main__':
    try:
        delete_dax_table(config._TABLE_INGEST)
        delete_dax_table(config._TABLE_MODELS)
    except ClientError:
        pass

    creandoTablaIngesta(config._TABLE_INGEST)
    creandoTablaModelos(config._TABLE_MODELS)

    
    
