# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Composite connector for heterogeneous inbound/outbound edges.

Middle stages may receive on one transport (e.g. Mooncake cross-node) and
send on another (e.g. SharedMemory when colocated with the next stage).
``merge_stage_connector_specs`` encodes that as a CompositeOmniConnector
wrapping a get-side connector and a put-side connector.
"""

from typing import Any

from ..utils.logging import get_connector_logger
from .base import OmniConnectorBase

logger = get_connector_logger(__name__)


class CompositeOmniConnector(OmniConnectorBase):
    """Delegates ``get`` / ``put`` to distinct underlying connectors."""

    def __init__(self, config: dict[str, Any]):
        from ..factory import OmniConnectorFactory
        from ..utils.config import ConnectorSpec

        self.config = config
        self._get_connector: OmniConnectorBase | None = None
        self._put_connector: OmniConnectorBase | None = None

        stage_id = config.get("stage_id")
        get_spec = config.get("get_connector")
        put_spec = config.get("put_connector")

        if get_spec:
            self._get_connector = OmniConnectorFactory.create_connector(
                self._child_spec(get_spec, stage_id)
            )
        if put_spec:
            self._put_connector = OmniConnectorFactory.create_connector(
                self._child_spec(put_spec, stage_id)
            )

        if self._get_connector is None and self._put_connector is None:
            raise ValueError(
                "CompositeOmniConnector requires at least one of "
                "get_connector / put_connector"
            )

        # Prefer put-side for send-path raw-data decisions; fall back to get.
        put_raw = bool(getattr(self._put_connector, "supports_raw_data", False))
        get_raw = bool(getattr(self._get_connector, "supports_raw_data", False))
        if self._put_connector is not None and self._get_connector is not None:
            self.supports_raw_data = put_raw and get_raw
        elif self._put_connector is not None:
            self.supports_raw_data = put_raw
        else:
            self.supports_raw_data = get_raw

        logger.info(
            "CompositeOmniConnector: get=%s put=%s",
            type(self._get_connector).__name__ if self._get_connector else None,
            type(self._put_connector).__name__ if self._put_connector else None,
        )

    @staticmethod
    def _child_spec(spec: dict[str, Any], stage_id: Any):
        from ..utils.config import ConnectorSpec

        name = spec.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Composite child connector missing name")
        extra = dict(spec.get("extra") or {})
        if stage_id is not None:
            extra.setdefault("stage_id", stage_id)
        return ConnectorSpec(name=name.strip(), extra=extra)

    def put(
        self, from_stage: str, to_stage: str, put_key: str, data: Any
    ) -> tuple[bool, int, dict[str, Any] | None]:
        if self._put_connector is None:
            raise RuntimeError("CompositeOmniConnector has no put_connector")
        return self._put_connector.put(from_stage, to_stage, put_key, data)

    def get(
        self,
        from_stage: str,
        to_stage: str,
        get_key: str,
        metadata: dict[str, Any] | None = None,
    ) -> tuple[Any, int] | None:
        if self._get_connector is None:
            raise RuntimeError("CompositeOmniConnector has no get_connector")
        return self._get_connector.get(from_stage, to_stage, get_key, metadata)

    def cleanup(self, request_id: str) -> None:
        if self._get_connector is not None:
            self._get_connector.cleanup(request_id)
        if self._put_connector is not None:
            self._put_connector.cleanup(request_id)

    def health(self) -> dict[str, Any]:
        return {
            "get": self._get_connector.health() if self._get_connector else None,
            "put": self._put_connector.health() if self._put_connector else None,
        }

    def close(self) -> None:
        if self._get_connector is not None:
            self._get_connector.close()
        if self._put_connector is not None:
            self._put_connector.close()
