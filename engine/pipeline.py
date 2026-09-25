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

    A ValueError because it means the caller's request was bad — the API
    turns this into a 400, not a 500.
    """


class StageFailedError(RuntimeError):
    """A stage ran and failed. Carries the stage name for attribution."""

    def __init__(self, stage: str, message: str):
        self.stage = stage
        super().__init__(f"stage '{stage}' failed: {message}")


def available_stages() -> typing.List[str]:
    """Every stage name the engine knows about.

    Reads directory entries only — nothing is imported, so this is
    microseconds and cannot pull in torch. This is the function the API
    uses to answer "what can this engine do?".
    """
    names = []

    for info in pkgutil.iter_modules(path=[STAGES_DIR]):
        if info.ispkg:
            # skip packages, we only want modules
            logging.debug(f"Skipping package: {info.name}")
            continue
        names.append(info.name)

    return sorted(names)


def validate(stages: list) -> None:
    """Raise if any requested stage name doesn't exist.

    Extra modules in the stages directory that nobody asked for are fine —
    that's capability we're not using today. This only checks the other
    direction: does everything requested exist?

    Imports nothing, so the API can call it before enqueueing a job.
    """
    known = set(available_stages())
    unknown = [stage for stage in stages if stage not in known]

    if unknown:
        raise UnknownStageError(
            f"unknown stages: {unknown}. available: {sorted(known)}"
        )


def load_stages(stages: list) -> typing.Dict[str, typing.Callable]:
    """Import each requested stage module and pull out its run() function.

    Keyed by stage name, not by position, so a stage can never end up
    running under the wrong name. Call validate() first — this assumes the
    names are already known to be good.

    This is the expensive half: importing a stage module runs its top-level
    code, which is why stage modules must keep heavy imports (torch,
    librosa) inside run() rather than at module level.
    """
    loaded = {}

    for stage in stages:
        if stage in loaded:
            # already imported; a duplicate in the list is legal
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

    # attributes not passed in constructor
    overall_progress: float = 0.0
    current_stage: CurrentStage = None
    previous_stage: str = None
    stage_index: int = 0
    data: dict = field(default_factory=dict) # way to safely initalize dict 
    on_progress: collections.abc.Callable = None

    def report(self, fraction: float) -> None:
        """Record progress within the current stage and tell the caller.

        Reports facts only — which stage, where it sits in the list, how far
        into it we are. It never blends them into one overall percentage;
        the caller decides how to present that.
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
    """Run the notebook's stages in order, sharing one notebook between them.

    Stages mutate the notebook and return nothing. Dependencies are
    implicit: a stage checks its own inputs and raises if what it needs
    isn't there, so nothing here validates ordering.
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