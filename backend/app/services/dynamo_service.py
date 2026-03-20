import logging
import time
from typing import Dict, List, Optional
from datetime import datetime, timezone
from decimal import Decimal

import boto3
from botocore.exceptions import ClientError

from app.config import get_settings

logger = logging.getLogger(__name__)

_table = None


def _get_table():
    global _table
    if _table is None:
        settings = get_settings()
        client_kwargs = {"region_name": settings.AWS_REGION}
        if settings.AWS_ACCESS_KEY_ID and settings.AWS_SECRET_ACCESS_KEY:
            client_kwargs["aws_access_key_id"] = settings.AWS_ACCESS_KEY_ID
            client_kwargs["aws_secret_access_key"] = settings.AWS_SECRET_ACCESS_KEY
        dynamodb = boto3.resource("dynamodb", **client_kwargs)
        _table = dynamodb.Table(settings.DYNAMODB_TABLE_NAME)
        try:
            _table.load()
            logger.info(f"DynamoDB table '{settings.DYNAMODB_TABLE_NAME}' connected")
        except ClientError as e:
            if e.response["Error"]["Code"] == "ResourceNotFoundException":
                logger.warning(f"DynamoDB table '{settings.DYNAMODB_TABLE_NAME}' not found, creating it")
                dynamodb.create_table(
                    TableName=settings.DYNAMODB_TABLE_NAME,
                    KeySchema=[
                        {"AttributeName": "user_id", "KeyType": "HASH"},
                        {"AttributeName": "timestamp", "KeyType": "RANGE"},
                    ],
                    AttributeDefinitions=[
                        {"AttributeName": "user_id", "AttributeType": "S"},
                        {"AttributeName": "timestamp", "AttributeType": "S"},
                    ],
                    BillingMode="PAY_PER_REQUEST",
                )
                _table = dynamodb.Table(settings.DYNAMODB_TABLE_NAME)
                _table.wait_until_exists()
                logger.info(f"DynamoDB table '{settings.DYNAMODB_TABLE_NAME}' created")
            else:
                logger.error(f"DynamoDB error: {e}")
                raise
    return _table


def _float_to_decimal(obj):
    if isinstance(obj, float):
        return Decimal(str(round(obj, 6)))
    if isinstance(obj, dict):
        return {k: _float_to_decimal(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_float_to_decimal(i) for i in obj]
    return obj


def _decimal_to_float(obj):
    if isinstance(obj, Decimal):
        return float(obj)
    if isinstance(obj, dict):
        return {k: _decimal_to_float(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_decimal_to_float(i) for i in obj]
    return obj


def save_interaction(
    user_id: str,
    dominant_emotion: str,
    emotion_vector: Dict[str, float],
    confidence: float,
    message: str = "",
    reply: str = "",
) -> bool:
    try:
        table = _get_table()
        now = datetime.now(timezone.utc).isoformat()
        item = {
            "user_id": user_id,
            "timestamp": now,
            "dominant_emotion": dominant_emotion,
            "emotion_vector": _float_to_decimal(emotion_vector),
            "confidence": _float_to_decimal(confidence),
            "message": message,
            "reply": reply,
        }
        table.put_item(Item=item)
        logger.info(f"Saved interaction for user {user_id[:8]}... emotion={dominant_emotion}")
        return True
    except Exception as e:
        logger.error(f"Failed to save interaction: {e}")
        return False


def fetch_emotion_history(
    user_id: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> List[Dict]:
    try:
        table = _get_table()
        if start_date and end_date:
            from boto3.dynamodb.conditions import Key
            response = table.query(
                KeyConditionExpression=Key("user_id").eq(user_id)
                & Key("timestamp").between(start_date, end_date),
            )
        elif start_date:
            from boto3.dynamodb.conditions import Key
            response = table.query(
                KeyConditionExpression=Key("user_id").eq(user_id)
                & Key("timestamp").gte(start_date),
            )
        else:
            from boto3.dynamodb.conditions import Key
            response = table.query(
                KeyConditionExpression=Key("user_id").eq(user_id),
            )
        items = response.get("Items", [])
        items = [_decimal_to_float(item) for item in items]
        items.sort(key=lambda x: x.get("timestamp", ""))
        logger.info(f"Fetched {len(items)} records for user {user_id[:8]}...")
        return items
    except Exception as e:
        logger.error(f"Failed to fetch emotion history: {e}")
        return []
