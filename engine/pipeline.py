import pkgutil
import importlib
import typing
import logging
import sys
import inspect

from dataclasses import dataclass
from pathlib import Path

def print_module_list(Stages: list) -> typing.List[str]:
    modules = []

    for stage in Stages:
        logging.info(f"Searching for modules in stage: {stage}")

        for info in pkgutil.iter_modules(path=[Path(__file__).parent / "stages"]):
            if info.ispkg:
                # skip packages, we only want modules
                logging.info(f"Skipping package: {info.name}")
                continue

            if info.name == stage:
                logging.info(f"Found module: {info.name}")
                modules.append(info.name)

    return f"{modules}"

def load_module_list(Stages: list) -> typing.List[typing.Any]:
    amount_of_stages = 0 # used to see if modules were found for all stages, if not 
                         # not a real stage 
    modules = []         # used to store loaded module name, and list of functions (hooks)

    for stage in Stages:
        logging.info(f"Loading modules from stage: {stage}")

        for info in pkgutil.iter_modules(path=[Path(__file__).parent / "stages"]):
            if info.ispkg:
                # skip packages, we only want modules
                logging.info(f"Skipping package: {info.name}")
                continue

            if info.name == stage:
                logging.info(f"Found module: {info.name}")
                module = importlib.import_module(f"stages.{info.name}")

                # tuple of (module_name, list_of_functions) to be stored in modules list
                functions = filter(lambda x: x[0] == "run", inspect.getmembers(module, inspect.isfunction))
                modules.append((module.__name__, functions))
                logging.info(f"Loaded module: {module.__name__}")
                amount_of_stages += 1

        logging.info(f"Total modules loaded for stage '{stage}': {len(modules)}")

    if amount_of_stages != len(Stages):
        raise RuntimeError(f"Some stages were not found in the stages directory. Found stages: {amount_of_stages}, Requested stages: {Stages}")

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
    Notebook.modules = load_module_list(Notebook.stages)
    Notebook.amount_of_stages = len(Notebook.stages)
    Notebook.data = {}

    for index, stage in enumerate(Notebook.stages):
        Notebook.current_stage = CurrentStage(name=stage, progress=0.0, status="Not Started", hooks=Notebook.modules[index][1])
        logging.info(f"{stage}'s status: {Notebook.current_stage['status']}")
        logging.info(f"Executing stage: {stage}")

        for hook in Notebook.current_stage["hooks"]:
            logging.info(f"Executing hook: {hook[0]}")
            hook[1](Notebook)  # Execute the hook function for the current stage
            logging.info(f"Completed hook: {hook[0]}")

        # def stage(hook):

        #     # @Stage is the decorator used for each hook function
        #     def wrapper(Notebook):
        #         logging.info(f"Executing hook: {hook[0]}")
        #         hook[1](Notebook)
        #         logging.info(f"Completed hook: {hook[0]}")

        #     return wrapper

        # @stage
        # def execute_stage(Notebook):
        #     for hook in Notebook.current_stage["hooks"]:
        #         stage(hook)(Notebook)

        if Notebook.current_stage["status"] != "Completed":
            raise RuntimeError(f"Stage '{stage}' did not complete successfully. Current status: {Notebook.current_stage['status']}")

        logging.info(f"Completed stage: {stage}")
        logging.info(f"{stage}'s status: {Notebook.current_stage['status']}")
        Notebook.overall_progress += 1 / Notebook.amount_of_stages
        Notebook.previous_stage = stage
        index += 1
    
    logging.info("Pipeline execution completed")