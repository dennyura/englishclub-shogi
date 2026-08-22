"""Google Drive synchronization helpers for Kaggle training."""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any


class GoogleDriveSync:
    """Upload adopted checkpoints and append completed-training logs to Drive."""

    def __init__(self, service: Any, folder_id: str) -> None:
        self._service = service
        self._folder_id = folder_id

    @classmethod
    def from_service_account_json(
        cls,
        service_account_json: str,
        folder_id: str,
    ) -> GoogleDriveSync:
        """Create a Drive client from a service-account JSON string."""
        from google.oauth2 import service_account
        from googleapiclient.discovery import build

        credentials = service_account.Credentials.from_service_account_info(
            json.loads(service_account_json),
            scopes=["https://www.googleapis.com/auth/drive"],
        )
        return cls(build("drive", "v3", credentials=credentials), folder_id)

    @classmethod
    def from_kaggle_secret(cls, secret_name: str, folder_id: str) -> GoogleDriveSync:
        """Create a Drive client using a JSON service-account Kaggle Secret."""
        from kaggle_secrets import UserSecretsClient

        secret = UserSecretsClient().get_secret(secret_name)
        return cls.from_service_account_json(secret, folder_id)

    def download_checkpoint(self, local_path: Path, drive_name: str | None = None) -> bool:
        """Download a Drive checkpoint when it exists and return whether it was found."""
        from googleapiclient.http import MediaIoBaseDownload

        file_id = self._find_file(drive_name or local_path.name)
        if file_id is None:
            return False

        local_path.parent.mkdir(parents=True, exist_ok=True)
        request = self._service.files().get_media(fileId=file_id)
        with local_path.open("wb") as output:
            downloader = MediaIoBaseDownload(output, request)
            done = False
            while not done:
                _, done = downloader.next_chunk()
        return True

    def upload_checkpoint(self, local_path: Path) -> None:
        """Create or overwrite the checkpoint with the same name in Drive."""
        from googleapiclient.http import MediaFileUpload

        file_id = self._find_file(local_path.name)
        media = MediaFileUpload(str(local_path), mimetype="application/octet-stream")
        if file_id is None:
            self._service.files().create(
                body={"name": local_path.name, "parents": [self._folder_id]},
                media_body=media,
                fields="id",
            ).execute()
        else:
            self._service.files().update(fileId=file_id, media_body=media).execute()

    def append_log(self, log_text: str, log_name: str = "log.txt") -> None:
        """Append text to a Drive log, creating it when it does not exist."""
        from googleapiclient.http import MediaIoBaseUpload

        file_id = self._find_file(log_name)
        content = self._download_text(file_id) if file_id is not None else ""
        if content and not content.endswith("\n"):
            content += "\n"
        content += log_text
        media = MediaIoBaseUpload(
            io.BytesIO(content.encode("utf-8")),
            mimetype="text/plain",
            resumable=False,
        )
        if file_id is None:
            self._service.files().create(
                body={"name": log_name, "parents": [self._folder_id]},
                media_body=media,
                fields="id",
            ).execute()
        else:
            self._service.files().update(fileId=file_id, media_body=media).execute()

    def _find_file(self, name: str) -> str | None:
        escaped_name = name.replace("'", "\\'")
        query = (
            f"name = '{escaped_name}' "
            f"and '{self._folder_id}' in parents and trashed = false"
        )
        response = self._service.files().list(
            q=query,
            spaces="drive",
            fields="files(id, name)",
            pageSize=1,
        ).execute()
        files = response.get("files", [])
        return files[0]["id"] if files else None

    def _download_text(self, file_id: str) -> str:
        response = self._service.files().get_media(fileId=file_id).execute()
        if isinstance(response, bytes):
            return response.decode("utf-8")
        return bytes(response).decode("utf-8")
