from __future__ import annotations

import json
import os
import unittest
from unittest.mock import patch

from codex_modal.custom_deploy import (
    CUSTOM_APP_CONFIG_ENV,
    build_custom_deployment,
    custom_app_name,
    custom_app_base_url,
    gpu_count,
    parse_sglang_args,
)
from codex_modal.options import load_presets, parse_arguments, resolve_model


class CustomDeploymentTests(unittest.TestCase):
    def test_builds_hash_reuse_deployment_without_embedding_hf_token(self) -> None:
        arguments = [
            "--modal-self-managed",
            "--modal-model",
            "base/model",
            "--modal-model-revision",
            "a" * 40,
            "--modal-custom-hf-repo",
            "org/fine-tune",
            "--modal-custom-hf-revision",
            "b" * 40,
            "--modal-base-volume",
            "base-model-cache",
            "--modal-base-volume-path",
            "/snapshots/base",
            "--modal-gpu",
            "B200:2",
            "--modal-sglang-arg",
            "--trust-remote-code",
            "--modal-sglang-arg",
            "--reasoning-parser=example",
            "--modal-hf-token-env",
            "TEST_PRIVATE_HF_TOKEN",
        ]
        with patch.dict(os.environ, {"TEST_PRIVATE_HF_TOKEN": "hf-secret-value"}):
            options = parse_arguments(arguments)
            resolved = resolve_model(options, load_presets())
            deployment = build_custom_deployment(options, resolved, "codex-custom-test")
            environment = deployment.deployment_environment()

        self.assertEqual(deployment.source_repo, "org/fine-tune")
        self.assertEqual(deployment.base_repo, "base/model")
        self.assertEqual(deployment.tensor_parallel_size, 2)
        self.assertEqual(deployment.max_containers, 1)
        self.assertEqual(deployment.scaledown_window, 300)
        self.assertEqual(deployment.server_args["--trust-remote-code"], "")
        self.assertTrue(deployment.volume_name.startswith("cm-"))
        self.assertNotIn("hf-secret-value", environment[CUSTOM_APP_CONFIG_ENV])
        self.assertEqual(environment["MODAL_IMAGE_BUILDER_VERSION"], "2025.06")
        document = json.loads(environment[CUSTOM_APP_CONFIG_ENV])
        self.assertEqual(document["hf_token_env"], "TEST_PRIVATE_HF_TOKEN")

    def test_standalone_repo_does_not_require_catalog_or_base_volume(self) -> None:
        options = parse_arguments(
            [
                "--modal-self-managed",
                "--modal-model",
                "org/non-catalog-model",
                "--modal-model-revision",
                "release",
                "--modal-gpu",
                "H100",
            ]
        )
        deployment = build_custom_deployment(
            options, resolve_model(options, load_presets()), "codex-standalone-test"
        )
        self.assertEqual(deployment.source_repo, "org/non-catalog-model")
        self.assertEqual(deployment.source_revision, "release")
        self.assertIsNone(deployment.base_volume)

    def test_reserved_sglang_args_and_invalid_gpu_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            parse_sglang_args(["--model-path=/escape"])
        with self.assertRaises(ValueError):
            parse_sglang_args(["--foo=one", "--foo=two"])
        with self.assertRaises(ValueError):
            gpu_count("B200:99")

    def test_custom_app_url_matches_modal_server_naming(self) -> None:
        self.assertEqual(
            custom_app_base_url("workspace", "codex-test", "us-west"),
            "https://workspace--codex-test-server.us-west.modal.direct/v1",
        )

    def test_long_app_name_is_shortened_for_full_modal_dns_label(self) -> None:
        requested = "codex-deepseek-v4-flash-0731-abliterated-e9a2011a"
        shortened = custom_app_name(requested, "sshh12")
        self.assertLess(len(shortened), len(requested))
        self.assertLessEqual(len(f"sshh12--{shortened}-server"), 63)
        self.assertEqual(shortened, custom_app_name(requested, "sshh12"))
        self.assertIn(shortened, custom_app_base_url("sshh12", shortened, "us-west"))

    def test_overlong_custom_app_url_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            custom_app_base_url("workspace", "x" * 63, "us-west")

    def test_default_runtime_is_sglang(self) -> None:
        options = parse_arguments(
            [
                "--modal-self-managed",
                "--modal-model",
                "org/model",
                "--modal-model-revision",
                "main",
                "--modal-gpu",
                "H100",
            ]
        )
        deployment = build_custom_deployment(
            options, resolve_model(options, load_presets()), "codex-sglang-default"
        )
        self.assertEqual(deployment.runtime, "sglang")
        self.assertIsNone(deployment.download_subdir)
        self.assertIn("sglang", deployment.sglang_image)

    def test_llamacpp_runtime_uses_cuda_base_and_subdir(self) -> None:
        options = parse_arguments(
            [
                "--modal-self-managed",
                "--modal-model",
                "org/model-GGUF",
                "--modal-model-revision",
                "main",
                "--modal-gpu",
                "H200:2",
                "--modal-runtime",
                "llamacpp",
                "--modal-download-subdir",
                "Mix-256G",
                "--modal-sglang-arg",
                "--ctx-size=32768",
                "--modal-sglang-arg",
                "--no-reasoning-preserve",
            ]
        )
        deployment = build_custom_deployment(
            options, resolve_model(options, load_presets()), "codex-gguf-test"
        )
        self.assertEqual(deployment.runtime, "llamacpp")
        self.assertEqual(deployment.download_subdir, "Mix-256G")
        self.assertEqual(deployment.sglang_image, "nvidia/cuda:12.8.0-devel-ubuntu22.04")
        self.assertEqual(deployment.server_args["--ctx-size"], "32768")
        self.assertEqual(deployment.server_args["--no-reasoning-preserve"], "")

    def test_llamacpp_requires_download_subdir(self) -> None:
        options = parse_arguments(
            [
                "--modal-self-managed",
                "--modal-model",
                "org/model-GGUF",
                "--modal-model-revision",
                "main",
                "--modal-gpu",
                "H200:2",
                "--modal-runtime",
                "llamacpp",
            ]
        )
        with self.assertRaises((ValueError, RuntimeError)):
            build_custom_deployment(
                options, resolve_model(options, load_presets()), "codex-gguf-nosubdir"
            )

    def test_llamacpp_reserved_args_are_rejected(self) -> None:
        options = parse_arguments(
            [
                "--modal-self-managed",
                "--modal-model",
                "org/model-GGUF",
                "--modal-model-revision",
                "main",
                "--modal-gpu",
                "H200:2",
                "--modal-runtime",
                "llamacpp",
                "--modal-download-subdir",
                "Mix-256G",
                "--modal-sglang-arg",
                "--n-gpu-layers=80",
            ]
        )
        with self.assertRaises(ValueError):
            build_custom_deployment(
                options, resolve_model(options, load_presets()), "codex-gguf-reserved"
            )

    def test_unknown_runtime_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            parse_arguments(
                [
                    "--modal-self-managed",
                    "--modal-model",
                    "org/model",
                    "--modal-model-revision",
                    "main",
                    "--modal-gpu",
                    "H100",
                    "--modal-runtime",
                    "tensorrt",
                ]
            )


if __name__ == "__main__":
    unittest.main()
