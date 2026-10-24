from __future__ import annotations

import json
import unittest

from codex_modal.catalog_deploy import _load_models
from codex_modal.options import load_presets
from codex_modal.paths import CATALOG_PATH


class UnifiedCatalogTests(unittest.TestCase):
    def test_catalog_file_has_both_sections(self) -> None:
        document = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
        self.assertIn("modal_managed", document)
        self.assertIn("self_managed", document)
        self.assertIsInstance(document["modal_managed"]["presets"], dict)
        self.assertIsInstance(document["self_managed"]["models"], list)

    def test_load_presets_reads_modal_managed_section(self) -> None:
        presets = load_presets()
        self.assertIn("default", presets)
        self.assertIn(presets["default"], presets["presets"])

    def test_catalog_deploy_reads_self_managed_models(self) -> None:
        models = _load_models(CATALOG_PATH)
        self.assertTrue(models)
        names = {entry["name"] for entry in models}
        self.assertIn("deepseek-v4.1-flash-abliterated-gguf", names)

    def test_explicit_bare_array_still_supported(self) -> None:
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "custom.json"
            path.write_text(json.dumps([{"name": "x"}]), encoding="utf-8")
            self.assertEqual(_load_models(path), [{"name": "x"}])


if __name__ == "__main__":
    unittest.main()
