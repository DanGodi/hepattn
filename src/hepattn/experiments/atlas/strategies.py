from datetime import timedelta

from lightning.pytorch.strategies import DDPStrategy


class LongTimeoutDDPStrategy(DDPStrategy):
    def __init__(self, *args, timeout_minutes: int = 120, **kwargs):
        super().__init__(*args, timeout=timedelta(minutes=timeout_minutes), **kwargs)
