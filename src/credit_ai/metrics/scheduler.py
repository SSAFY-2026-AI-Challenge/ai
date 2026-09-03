from __future__ import annotations

import asyncio
import logging

from credit_ai.config import Settings
from credit_ai.metrics.refresh import refresh_economic_metrics

logger = logging.getLogger("credit_ai.metrics")


async def run_metric_scheduler(settings: Settings) -> None:
    interval = max(1, settings.metric_interval_seconds)
    loop = asyncio.get_running_loop()
    logger.info("economic_metric 갱신 주기 %s초", interval)
    while True:
        try:
            await loop.run_in_executor(None, refresh_economic_metrics, settings)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("economic_metric 갱신 실패")
        await asyncio.sleep(interval)
