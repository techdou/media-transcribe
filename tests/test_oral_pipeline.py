import importlib.util
import io
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "oral_pipeline.py"
SPEC = importlib.util.spec_from_file_location("oral_pipeline", SCRIPT)
pipeline = importlib.util.module_from_spec(SPEC)
# 不加载技能目录或调用目录中的真实 .env。
with patch.dict(os.environ, {}, clear=True), patch.object(Path, "is_file", return_value=False):
    SPEC.loader.exec_module(pipeline)


class OralPipelineTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name).resolve()
        self.addCleanup(os.chdir, Path.cwd())
        self.addCleanup(setattr, sys, "path", sys.path[:])
        os.chdir(self.root)
        self.engine = self.root / "engine"
        bin_dir = self.engine / "qwen_asr_gguf/inference/bin"
        bin_dir.mkdir(parents=True)
        for name in ("ggml.dll", "ggml-base.dll", "llama.dll"):
            (bin_dir / name).touch()
        (self.engine / "chinese_itn.py").touch()
        self.models = self.root / "models"
        self.models.mkdir()
        for family in ("asr", "aligner"):
            for suffix in ("encoder_frontend.onnx", "encoder_backend.onnx", "llm.gguf"):
                (self.models / f"qwen3_{family}_{suffix}").touch()
        self.cli_paths = ["--engine-dir", "engine", "--model-dir", "models"]

    def test_relative_output_rejects_duplicate_stems_before_engine_load(self):
        for name in ("a/demo.wav", "b/demo.wav", "b/demo.mp3"):
            path = self.root / name
            path.parent.mkdir(exist_ok=True)
            path.touch()
        for second in ("b/demo.wav", "b/demo.mp3"):
            with self.subTest(second=second):
                stderr = io.StringIO()
                with redirect_stderr(stderr), self.assertRaises(SystemExit) as result:
                    pipeline.main(["a/demo.wav", second, *self.cli_paths, "--out-dir", "out"])
                self.assertEqual(result.exception.code, 2)
                for marker in ("输出同名冲突", str(self.root / "a/demo.wav"), str(self.root / second)):
                    self.assertIn(marker, stderr.getvalue())
        self.assertEqual(Path.cwd(), self.root)
        self.assertFalse((self.root / "out").exists())
        self.assertFalse((self.engine / "out").exists())

    def test_relative_model_and_output_paths_survive_chdir(self):
        (self.root / "demo.wav").touch()
        fake_engine = Mock()
        fake_engine.transcribe.return_value = SimpleNamespace(text="demo", alignment=[])
        inference = ModuleType("qwen_asr_gguf.inference")
        inference.ASREngineConfig = inference.AlignerConfig = SimpleNamespace
        inference.QwenASREngine = Mock(return_value=fake_engine)
        inference.exporters = SimpleNamespace(
            export_to_txt=lambda path, res: Path(path).write_text(res.text, encoding="utf-8")
        )
        with patch.dict(sys.modules, {inference.__name__: inference}), redirect_stdout(io.StringIO()):
            pipeline.main(["demo.wav", *self.cli_paths, "--provider", "CPU", "--out-dir", "out"])
        config = inference.QwenASREngine.call_args.kwargs["config"]
        self.assertEqual(Path(config.model_dir), self.models)
        self.assertEqual(config.onnx_provider, "CPU")
        self.assertFalse(config.llm_use_gpu)
        self.assertFalse(config.align_config.llm_use_gpu)
        self.assertEqual((self.root / "out/demo.txt").read_text(encoding="utf-8"), "demo")
        self.assertTrue((self.root / "out/demo_timed.md").is_file())
        self.assertFalse((self.engine / "out").exists())
        fake_engine.shutdown.assert_called_once()

    def test_relative_output_does_not_replace_existing_artifact(self):
        out = self.root / "out"
        out.mkdir()
        artifact = out / "demo.txt"
        artifact.write_text("existing", encoding="utf-8")
        (self.root / "demo.wav").touch()
        stderr = io.StringIO()
        with redirect_stderr(stderr), self.assertRaises(SystemExit) as result:
            pipeline.main(["demo.wav", *self.cli_paths, "--out-dir", "out"])
        self.assertEqual(result.exception.code, 2)
        self.assertIn("输出已存在", stderr.getvalue())
        self.assertEqual(artifact.read_text(encoding="utf-8"), "existing")

    def test_exclusive_write_rejects_existing_target_on_concurrent_race(self):
        out = self.root / "out"
        out.mkdir()
        existing = out / "direct_timed.md"
        existing.write_text("keep-direct", encoding="utf-8")
        with self.assertRaises(FileExistsError):
            pipeline.write_timed_md(str(existing), "demo.wav", 1.0, [], "new", exclusive=True)
        self.assertEqual(existing.read_text(encoding="utf-8"), "keep-direct")

        # Simulate a concurrent process creating out/demo.txt AFTER plan_outputs passed.
        (self.root / "demo.wav").touch()
        raced_txt = out / "demo.txt"
        raced_md = out / "demo_timed.md"

        def concurrent_transcribe(**kwargs):
            raced_txt.write_text("from-other-process", encoding="utf-8")
            return SimpleNamespace(text="mine", alignment=[])

        fake_engine = Mock()
        fake_engine.transcribe.side_effect = concurrent_transcribe
        inference = ModuleType("qwen_asr_gguf.inference")
        inference.ASREngineConfig = inference.AlignerConfig = SimpleNamespace
        inference.QwenASREngine = Mock(return_value=fake_engine)
        inference.exporters = SimpleNamespace(
            export_to_txt=lambda path, res: Path(path).write_text(res.text, encoding="utf-8")
        )
        stderr = io.StringIO()
        with (patch.dict(sys.modules, {inference.__name__: inference}),
              redirect_stdout(io.StringIO()), redirect_stderr(stderr),
              self.assertRaises(SystemExit) as result):
            pipeline.main(["demo.wav", *self.cli_paths, "--provider", "CPU", "--out-dir", "out"])
        self.assertEqual(result.exception.code, 2)
        self.assertIn("输出已存在，拒绝覆盖", stderr.getvalue())
        self.assertEqual(raced_txt.read_text(encoding="utf-8"), "from-other-process")
        self.assertFalse(raced_md.exists())
        fake_engine.shutdown.assert_called_once()

    def test_doctor_checks_actual_provider_and_device(self):
        modules = {name: ModuleType(name) for name in
                   ("onnxruntime", "gguf", "srt", "soundfile", "typer", "rich", "nagisa", "soynlp")}
        modules["onnxruntime"].get_available_providers = lambda: ["CPUExecutionProvider", "CUDAExecutionProvider"]
        for options, code, provider, needs_gpu in (
            (["--provider", "CPU"], 0, "CPU", False),
            (["--provider", "CUDA", "--device", "CPU"], 0, "CUDA", False),
            (["--provider", "CPU", "--device", "gpu"], 1, "CPU", True),
        ):
            with self.subTest(options=options):
                stdout = io.StringIO()
                with (patch.dict(sys.modules, modules), patch.object(pipeline.shutil, "which", return_value="ffmpeg"),
                      redirect_stdout(stdout), self.assertRaises(SystemExit) as result):
                    pipeline.main(["--doctor", *self.cli_paths, *options])
                self.assertEqual(result.exception.code, code, stdout.getvalue())
                self.assertIn(f"[PASS] ONNX {provider} provider", stdout.getvalue())
                self.assertNotIn("DML", stdout.getvalue())
                self.assertEqual("[FAIL] DLL GPU" in stdout.getvalue(), needs_gpu)


if __name__ == "__main__":
    unittest.main()
