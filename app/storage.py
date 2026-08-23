"""Object storage abstraction.

Uses Cloudflare R2 (S3-compatible, via boto3) when R2 credentials are
configured. Falls back to local disk storage otherwise (handy for local
dev/tests; not recommended for production on Railway since local disk is
ephemeral across deploys).

Either way, files are never publicly enumerable: R2 objects are private and
served via short-lived presigned URLs; local files are served through a
Flask/FastAPI route that requires a signed, expiring token.
"""

from __future__ import annotations

import hashlib
import hmac
import time
from pathlib import Path
from urllib.parse import quote

from app.config import Settings, get_settings


class Storage:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self._s3 = None
        if self.settings.uses_r2:
            import boto3
            from botocore.config import Config

            self._s3 = boto3.client(
                "s3",
                endpoint_url=self.settings.resolved_r2_endpoint,
                aws_access_key_id=self.settings.r2_access_key_id,
                aws_secret_access_key=self.settings.r2_secret_access_key,
                region_name="auto",
                config=Config(signature_version="s3v4"),
            )

    @property
    def backend(self) -> str:
        return "r2" if self._s3 else "local"

    def upload_file(self, local_path: Path, key: str, content_type: str = "application/octet-stream") -> None:
        if self._s3:
            self._s3.upload_file(
                str(local_path),
                self.settings.r2_bucket,
                key,
                ExtraArgs={"ContentType": content_type},
            )
        else:
            dest = self.settings.local_storage_path / key
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(local_path.read_bytes())

    def delete(self, key: str) -> None:
        if self._s3:
            self._s3.delete_object(Bucket=self.settings.r2_bucket, Key=key)
        else:
            dest = self.settings.local_storage_path / key
            dest.unlink(missing_ok=True)

    def delete_prefix(self, prefix: str) -> None:
        if self._s3:
            paginator = self._s3.get_paginator("list_objects_v2")
            for page in paginator.paginate(Bucket=self.settings.r2_bucket, Prefix=prefix):
                objs = [{"Key": o["Key"]} for o in page.get("Contents", [])]
                if objs:
                    self._s3.delete_objects(Bucket=self.settings.r2_bucket, Delete={"Objects": objs})
        else:
            base = self.settings.local_storage_path / prefix
            if base.exists():
                if base.is_dir():
                    import shutil

                    shutil.rmtree(base, ignore_errors=True)
                else:
                    # prefix might not map to an exact dir; remove any matching files
                    parent = base.parent
                    if parent.exists():
                        for p in parent.glob(f"{base.name}*"):
                            p.unlink(missing_ok=True)

    def signed_download_url(self, key: str, *, expires_in: int = 3600, filename: str | None = None) -> str:
        if self._s3:
            params = {"Bucket": self.settings.r2_bucket, "Key": key}
            if filename:
                params["ResponseContentDisposition"] = f'attachment; filename="{filename}"'
            return self._s3.generate_presigned_url("get_object", Params=params, ExpiresIn=expires_in)
        return _local_signed_url(self.settings, key, expires_in=expires_in, filename=filename)


def _sign(settings: Settings, key: str, expires_at: int) -> str:
    msg = f"{key}:{expires_at}".encode()
    return hmac.new(settings.secret_key.encode(), msg, hashlib.sha256).hexdigest()


def _local_signed_url(settings: Settings, key: str, *, expires_in: int, filename: str | None) -> str:
    expires_at = int(time.time()) + expires_in
    sig = _sign(settings, key, expires_at)
    q = f"?exp={expires_at}&sig={sig}"
    if filename:
        q += f"&filename={quote(filename)}"
    return f"{settings.base_url}/files/{quote(key)}{q}"


def verify_local_signature(settings: Settings, key: str, exp: int, sig: str) -> bool:
    if int(time.time()) > exp:
        return False
    expected = _sign(settings, key, exp)
    return hmac.compare_digest(expected, sig)


_storage: Storage | None = None


def get_storage() -> Storage:
    global _storage
    if _storage is None:
        _storage = Storage()
    return _storage
