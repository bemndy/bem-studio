import pkgutil
import importlib
import typing
import collections.abc
import logging

from dataclasses import dataclass, field
from pathlib import Path

STAGES_DIR = Path(__file__).parent / "stages"


class UnknownStageError(ValueError):
    """A requested stage name has no module in the stages directory.

    Subclasses ValueError so the API can map it to a 400 (bad request)
    rather than a 500.
    """


class StageFailedError(RuntimeError):
    """A stage ran and failed. `stage` holds the name of the failing stage."""

    def __init__(self, stage: str, message: str):
        self.stage = stage
        super().__init__(f"stage '{stage}' failed: {message}")


def available_stages() -> typing.List[str]:
    """Return the sorted names of all stage modules in the stages directory.

    Only reads directory entries; no stage module is imported, so this is
    cheap and never loads heavy dependencies like torch.
    """
    names = []

    for info in pkgutil.iter_modules(path=[STAGES_DIR]):
        if info.ispkg:
            # stages are single modules; ignore packages
            logging.debug(f"Skipping package: {info.name}")
            continue
        names.append(info.name)

    return sorted(names)


def validate(stages: list) -> None:
    """Raise UnknownStageError if any requested stage name doesn't exist.

    Stages that exist but weren't requested are ignored. Imports nothing,
    so it is safe to call before enqueueing a job.
    """
    known = set(available_stages())
    unknown = [stage for stage in stages if stage not in known]

    if unknown:
        raise UnknownStageError(
            f"unknown stages: {unknown}. available: {sorted(known)}"
        )


def load_stages(stages: list) -> typing.Dict[str, typing.Callable]:
    """Import each requested stage module and return its run() by stage name.

    Assumes the names have already passed validate(). Importing a stage
    runs its top-level code, so stage modules keep heavy imports (torch,
    librosa) inside run() rather than at module level.
    """
    loaded = {}

    for stage in stages:
        if stage in loaded:
            # duplicate stage names are allowed; import once
            continue

        logging.info(f"Loading stage: {stage}")
        module = importlib.import_module(f"stages.{stage}")

        entry = getattr(module, "run", None)
        if entry is None:
            raise UnknownStageError(
                f"stage module '{stage}' has no run() function"
            )

        loaded[stage] = entry
        logging.info(f"Loaded stage: {stage}")

    return loaded


class CurrentStage(typing.TypedDict):
    name: str
    progress: float
    status: str


@dataclass
class Notebook:
    stages: list
    source: str
    audio: str

    # runtime state, set by run_pipeline
    overall_progress: float = 0.0
    current_stage: CurrentStage = None
    previous_stage: str = None
    stage_index: int = 0
    data: dict = field(default_factory=dict)  # fresh dict per instance
    on_progress: collections.abc.Callable = None

    def report(self, fraction: float) -> None:
        """Record progress within the current stage and notify on_progress.

        Calls on_progress(name, stage_index, total_stages, fraction). Overall
        progress is left for the caller to compute.
        """
        self.current_stage["progress"] = fraction

        if self.on_progress:
            self.on_progress(
                self.current_stage["name"],
                self.stage_index,
                len(self.stages),
                fraction,
            )


def run_pipeline(notebook: Notebook, on_progress: typing.Callable = None) -> Notebook:
    """Run the notebook's stages in order, passing the same notebook to each.

    Stages mutate the notebook and return nothing. Stage ordering is not
    validated here; each stage checks its own inputs and raises if they
    are missing.
    """
    logging.info("Starting pipeline execution")

    if on_progress is not None:
        notebook.on_progress = on_progress

    validate(notebook.stages)
    loaded = load_stages(notebook.stages)

    total = len(notebook.stages)

    for index, stage in enumerate(notebook.stages):
        notebook.stage_index = index
        notebook.current_stage = CurrentStage(
            name=stage, progress=0.0, status="Not Started"
        )
        logging.info(f"Executing stage: {stage} ({index + 1}/{total})")

        notebook.report(0.0)

        try:
            loaded[stage](notebook)
        except StageFailedError:
            raise
        except Exception as exc:
            notebook.current_stage["status"] = "Failed"
            raise StageFailedError(stage, str(exc)) from exc

        if notebook.current_stage["status"] != "Completed":
            raise StageFailedError(
                stage,
                f"did not complete successfully. "
                f"status: {notebook.current_stage['status']}",
            )

        notebook.report(1.0)

        logging.info(f"Completed stage: {stage}")
        notebook.overall_progress += 1 / total
        notebook.previous_stage = stage

    logging.info("Pipeline execution completed")
    return notebook