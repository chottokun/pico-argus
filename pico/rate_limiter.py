import asyncio
import time
import logging
from collections import deque

logger = logging.getLogger(__name__)

class RPMLimiter:
    """ローカルLLM/VLMの呼び出し間隔と回数を制御し、サーマルスロットリングやリソース枯渇を防ぐレートリミッター。"""
    def __init__(self, max_rpm: int = 6):
        self.max_rpm = max_rpm
        self.timestamps = deque()
        self.lock = asyncio.Lock()

    async def acquire(self) -> None:
        """リクエスト上限に達している場合は空きが出るまで非同期に待機し、許可が下りたら履歴に記録します。"""
        while True:
            wait_time = 0.0
            async with self.lock:
                now = time.monotonic()
                # 60秒より古いタイムスタンプをスライディングウィンドウから除外
                while self.timestamps and now - self.timestamps[0] > 60.0:
                    self.timestamps.popleft()

                if len(self.timestamps) < self.max_rpm:
                    self.timestamps.append(now)
                    logger.debug(f"RPM Limiter: Request acquired. current window size: {len(self.timestamps)}/{self.max_rpm}")
                    break
                else:
                    # 最古のリクエストが期限切れになるまでの時間
                    oldest = self.timestamps[0]
                    wait_time = 60.0 - (now - oldest)

            if wait_time > 0:
                logger.warning(f"⚠️ RPM Limit reached ({self.max_rpm} rpm). Waiting for {wait_time:.2f}s to prevent thermal throttling...")
                await asyncio.sleep(wait_time)
