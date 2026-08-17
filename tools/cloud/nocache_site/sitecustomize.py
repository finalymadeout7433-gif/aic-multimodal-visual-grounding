"""Runtime-only cache policy for the supervised Phase 1.8R cloud run.

This module is injected through PYTHONPATH.  It deliberately does not modify
the experiment fingerprint: model, loss, optimizer, data, and evaluation
semantics stay unchanged while resumable Teacher Bank writes are suppressed.
"""

from __future__ import annotations

import os
from pathlib import Path


if os.environ.get("AIC_DISABLE_TEACHER_BANK_CACHE") == "1":
    from aic_rgbtir.phase16 import RGBTeacherBank

    _original_save = RGBTeacherBank.save

    def _save_without_teacher_cache(self: RGBTeacherBank, path: Path | str) -> None:
        target = Path(path)
        if target.name == "teacher_bank.partial.pt":
            return
        _original_save(self, target)

    RGBTeacherBank.save = _save_without_teacher_cache


if os.environ.get("AIC_DISABLE_VALIDATION_CACHE") == "1":
    from aic_rgbtir.phase15 import Phase15Config

    _original_config_init = Phase15Config.__init__

    def _validation_config_without_cache(self: Phase15Config, *args, **kwargs) -> None:
        kwargs["cache_enabled"] = False
        _original_config_init(self, *args, **kwargs)

    Phase15Config.__init__ = _validation_config_without_cache
