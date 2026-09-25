import logging
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
logging.basicConfig(level=logging.INFO)

import pytest

import pipeline


def make_notebook(stages):
    return pipeline.Notebook(stages=stages, source="API", audio="./audio.wav")


class TestListingAndValidation:
    def test_available_stages_returns_a_list_of_names(self):
        names = pipeline.available_stages()
        assert isinstance(names, list)
        assert "a" in names
        assert "b" in names

    def test_validate_accepts_known_stages(self):
        assert pipeline.validate(["a", "b"]) is None

    def test_validate_allows_duplicates(self):
        assert pipeline.validate(["a", "a"]) is None

    def test_validate_reports_every_unknown_name(self):
        with pytest.raises(pipeline.UnknownStageError) as caught:
            pipeline.validate(["a", "tempoo", "nonsense"])

        message = str(caught.value)
        assert "tempoo" in message
        assert "nonsense" in message

    def test_unknown_stage_is_a_value_error(self):
        # UnknownStageError maps to a 400, StageFailedError to a 500
        assert issubclass(pipeline.UnknownStageError, ValueError)
        assert not issubclass(pipeline.UnknownStageError, RuntimeError)

    def test_listing_does_not_import_torch(self):
        # listing and validating must not import any stage module
        pipeline.available_stages()
        pipeline.validate(["a", "b"])
        assert "torch" not in sys.modules


class TestLoading:
    def test_load_stages_is_keyed_by_name(self):
        loaded = pipeline.load_stages(["a", "b"])
        assert set(loaded) == {"a", "b"}
        assert callable(loaded["a"])

    def test_load_stages_deduplicates(self):
        loaded = pipeline.load_stages(["a", "a"])
        assert list(loaded) == ["a"]


class TestRunPipeline:
    def test_stages_run_in_order_and_share_the_notebook(self):
        notebook = make_notebook(["a", "b"])
        pipeline.run_pipeline(notebook)

        # b read what a wrote, and doubled it
        assert notebook.data["a_data"] == pytest.approx([1.3, 5.4, 6.7])
        assert notebook.data["b_data"] == pytest.approx([2.6, 10.8, 13.4])

    def test_run_pipeline_returns_the_same_notebook(self):
        notebook = make_notebook(["a"])
        assert pipeline.run_pipeline(notebook) is notebook

    def test_bookkeeping_after_a_full_run(self):
        notebook = make_notebook(["a", "b"])
        pipeline.run_pipeline(notebook)

        assert notebook.overall_progress == pytest.approx(1.0)
        assert notebook.previous_stage == "b"
        assert notebook.current_stage["status"] == "Completed"

    def test_empty_stage_list_does_nothing_and_does_not_divide_by_zero(self):
        notebook = make_notebook([])
        pipeline.run_pipeline(notebook)

        assert notebook.data == {}
        assert notebook.overall_progress == 0.0
        assert notebook.previous_stage is None

    def test_unknown_stage_raises_before_any_stage_runs(self):
        notebook = make_notebook(["a", "nonsense"])

        with pytest.raises(pipeline.UnknownStageError):
            pipeline.run_pipeline(notebook)

        # "a" was listed first but must not have run
        assert notebook.data == {}

    def test_missing_prerequisite_fails_and_names_the_stage(self):
        notebook = make_notebook(["b"])

        with pytest.raises(pipeline.StageFailedError) as caught:
            pipeline.run_pipeline(notebook)

        assert caught.value.stage == "b"
        assert "a_data" in str(caught.value)


class TestProgress:
    def test_hook_receives_per_stage_facts(self):
        seen = []
        notebook = make_notebook(["a", "b"])

        pipeline.run_pipeline(
            notebook,
            on_progress=lambda name, index, total, fraction: seen.append(
                (name, index, total, fraction)
            ),
        )

        names = [event[0] for event in seen]
        assert names[0] == "a"
        assert names[-1] == "b"

        # every event carries the stage position and within-stage fraction
        for name, index, total, fraction in seen:
            assert total == 2
            assert index in (0, 1)
            assert 0.0 <= fraction <= 1.0

        # each stage reports 0.0 on entry and 1.0 on exit
        a_fractions = [e[3] for e in seen if e[0] == "a"]
        assert a_fractions[0] == 0.0
        assert a_fractions[-1] == 1.0

    def test_no_hook_still_runs(self):
        notebook = make_notebook(["a"])
        pipeline.run_pipeline(notebook)
        assert notebook.data["a_data"]