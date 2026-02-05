from crewai.tools import BaseTool
from typing import Type
from pydantic import BaseModel, Field
import os
import requests


from typing import Any

class PushNotification(BaseModel):
    message: Any = Field(..., description="The message to be sent to the user.")

class PushNotificationTool(BaseTool):
    name: str = "push"
    description: str = "Send a push notification to the user."
    args_schema: Type[BaseModel] = PushNotification

    def _run(self, message: str = "", **kwargs) -> str:
        if not message:
            message = kwargs.get("message", "")

        if isinstance(message, dict):
            message = (
                message.get("message")
                or message.get("description")
                or message.get("value")
                or str(message)
            )

        message = str(message)

        pushover_user = os.getenv("PUSHOVER_USER")
        pushover_token = os.getenv("PUSHOVER_TOKEN")

        if not pushover_user or not pushover_token:
            return "Push notification skipped: missing PUSHOVER_USER or PUSHOVER_TOKEN"

        pushover_url = "https://api.pushover.net/1/messages.json"
        payload = {"user": pushover_user, "token": pushover_token, "message": message}

        r = requests.post(pushover_url, data=payload, timeout=10)
        return f"Push status={r.status_code}, response={r.text[:200]}"
