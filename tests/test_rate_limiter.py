import pytest
from unittest.mock import patch
from pico.rate_limiter import RPMLimiter

@pytest.mark.anyio
async def test_rpm_limiter_allows_under_limit():
    limiter = RPMLimiter(max_rpm=3)
    # 制限以下の回数であれば待機なしで通過できること
    await limiter.acquire()
    await limiter.acquire()
    await limiter.acquire()
    assert len(limiter.timestamps) == 3

@pytest.mark.anyio
async def test_rpm_limiter_waits_when_limit_exceeded():
    limiter = RPMLimiter(max_rpm=2)
    # 2回実行して上限に達した状態を作る
    limiter.timestamps.append(100.0)
    limiter.timestamps.append(101.0)
    
    # 3回目を呼ぶと、最古の100.0から60秒経過するまで待機が発生することを確認
    with patch("time.monotonic") as mock_time, patch("asyncio.sleep") as mock_sleep:
        # 1回目ループ判定: 102.0 (wait_time = 60.0 - (102.0 - 100.0) = 58.0)
        # 2回目ループ判定: 161.0 (100.0 が期限切れになり、無事追加される)
        mock_time.side_effect = [102.0, 161.0, 161.0]
        mock_sleep.return_value = None

        await limiter.acquire()

        # 58秒間の sleep が非同期で呼び出されていること
        mock_sleep.assert_called_once_with(58.0)
        # 最終的なタイムスタンプ個数は上限内に収まっていること
        assert len(limiter.timestamps) == 2
        assert limiter.timestamps[0] == 101.0
        assert limiter.timestamps[1] == 161.0
