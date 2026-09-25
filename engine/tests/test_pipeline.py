import logging
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
logging.basicConfig(level=logging.INFO)
import pipeline
import pytest

class TestPipeline:
    def test_print_module_list(self):
        assert pipeline.print_module_list(["a", "b"]) == "['a', 'b']"

    def test_run_pipeline_with_a_and_b(self):
        notebook = pipeline.Notebook(stages=["a", "b"], source="API", audio="./audio.wav")
        assert pipeline.run_pipeline(notebook) == None

    def test_run_pipeline_with_empty_stages(self):
        notebook = pipeline.Notebook(stages=[], source="API", audio="./audio.wav")
        assert pipeline.run_pipeline(notebook) == None

    def test_run_pipeline_with_b_only_raises_runtime_error(self):
        notebook = pipeline.Notebook(stages=["b"], source="API", audio="./audio.wav")
        with pytest.raises(RuntimeError):
            pipeline.run_pipeline(notebook)