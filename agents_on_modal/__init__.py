"""agents-on-modal: agent CLIs x sandboxes x open-weight models served on Modal.

    import agents_on_modal as aom
    model = aom.models.get("qwen3.8-27b")
    endpoint = aom.models.deploy(model)
    box = aom.sandbox.create("modal")
    run = box.start("codex", endpoint, "build a todo app and test it in the browser", goal=True)
    for event in run.events(follow=True):
        print(event)
"""

from __future__ import annotations

__version__ = "0.2.0"

from . import dynamic, harness, models, sandbox
from .config import AomError
# Re-export the control-plane classes at the top level: they live in ``dynamic`` but the obvious
# import guesses are ``agents_on_modal.volumes`` / ``.control`` (issue S9), so surface them here.
from .dynamic import ControlPlane, LocalDirControlPlane, VolumeControlPlane

__all__ = ["AomError", "ControlPlane", "LocalDirControlPlane", "VolumeControlPlane",
           "__version__", "dynamic", "harness", "models", "sandbox"]
