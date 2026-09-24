import pkgutil
import importlib
import typing
import logging
import sys
import inspect

from dataclasses import dataclass
from pathlib import Path

def load_modules_from_list(Stages: list) -> typing.List[typing.Any]:
    filtered_stages = []
    for stage in Stages:
        modules = []
        logging.info(f"Loading modules from stage: {stage}")

        for info in pkgutil.iter_modules(path=[Path(__file__).parent / "paths"]):
            if info.ispkg:
                logging.info(f"Skipping package: {info.name}")
                continue
            if info.name.startswith(stage):
                logging.info(f"Found module: {info.name}")
                module = importlib.import_module(f"paths.{info.name}")
                if "torch" in sys.modules:
                    logging.warning("Torch should not be imported in this context, but it was")
                    exit(1)
                module = (module, inspect.getmembers(module, inspect.isfunction))
                modules.append(module)
                logging.info(f"Loaded module: {module.__name__}")
                filtered_stages.append(stage)
            else:
                logging.info(f"Module {info.name} does not match stage {stage}, skipping")

        logging.info(f"Total modules loaded for stage '{stage}': {len(modules)}")

    return filtered_stages, modules

class CurrentStage(typing.typedDict):
    name: str
    progress: float
    status: str
    hooks: list = None 

@dataclass
class Notebook:
    stages: list
    source: str
    audio: str

    # attributes not passed in constructor
    overall_progress: float = 0.0
    amount_of_stages: int = len(stages)
    current_stage: CurrentStage = None
    previous_stage: str = None
    modules: list = None
    data: dict = None


def run_pipeline(Notebook: Notebook):
    logging.info("Starting pipeline execution")
    Notebook.stages, Notebook.modules = load_modules_from_list(Notebook.stages)

    for index, stage in enumerate(Notebook.stages):
        Notebook.current_stage = CurrentStage(name=stage, progress=0.0, status="Not Started", hooks=Notebook.modules[Notebook.modules[index][1]])
        logging.info(f"Executing stage: {stage}")

        for hook in Notebook.current_stage["hooks"]:
            hook(Notebook)
            # all implemented by hook should update the Notebook.current_stage["progress"] as they complete their tasks

        Notebook.current_stage["progress"] += 1 / Notebook.amount_of_stages
        logging.info(f"Completed stage: {stage}")
        Notebook.previous_stage = Notebook.current_stage["name"]
        index += 1
    
    logging.info("Pipeline execution completed")