import pkgutil
import importlib
import typing
import logging
import sys
import inspect

from dataclasses import dataclass
from pathlib import Path

def load_modules_from_list(Stages: list) -> typing.List[typing.Any]:
    amount_of_stages = 0 # used to see if modules were found for all stages, if not 
                         # not a real stage 
    modules = []         # used to store loaded module name, and list of functions (hooks)

    for stage in Stages:
        logging.info(f"Loading modules from stage: {stage}")

        for info in pkgutil.iter_modules(path=[Path(__file__).parent / "paths"]):
            if info.ispkg:
                # skip packages, we only want modules
                logging.info(f"Skipping package: {info.name}")
                continue

            if info.name.startswith(stage):
                logging.info(f"Found module: {info.name}")
                module = importlib.import_module(f"paths.{info.name}")

                # if module imports pytorch, which API cannot handle, exit program immediately at runtime
                if "torch" in sys.modules:
                    logging.warning("Torch should not be imported in this context, but it was")
                    exit(1)

                # tuple of (module_name, list_of_functions) to be stored in modules list
                modules.append((module.__name__, inspect.getmembers(module, inspect.isfunction)))
                logging.info(f"Loaded module: {module.__name__}")
                amount_of_stages += 1

        logging.info(f"Total modules loaded for stage '{stage}': {len(modules)}")

    if amount_of_stages != len(Stages):
        logging.error(f"Some stages were not found in the paths directory. Found stages: {amount_of_stages}, Requested stages: {Stages}")
        exit(1)

    return modules

class CurrentStage(typing.TypedDict):
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
    current_stage: CurrentStage = None
    previous_stage: str = None
    modules: list = None 
    data: dict = None

def run_pipeline(Notebook: Notebook):
    logging.info("Starting pipeline execution")
    Notebook.modules = load_modules_from_list(Notebook.stages)
    Notebook.amount_of_stages = len(Notebook.stages)
    Notebook.data = {}

    for index, stage in enumerate(Notebook.stages):
        Notebook.current_stage = CurrentStage(name=stage, progress=0.0, status="Not Started", hooks=Notebook.modules[index][1])
        logging.info(f"Executing stage: {stage}")

        for hook in Notebook.current_stage["hooks"]:
            hook[1](Notebook)
            # all implemented by hook should update the Notebook.current_stage["progress"] as they complete their tasks

        Notebook.current_stage["progress"] += 1 / Notebook.amount_of_stages
        logging.info(f"Completed stage: {stage}")
        Notebook.previous_stage = Notebook.current_stage["name"]
        index += 1
    
    logging.info("Pipeline execution completed")