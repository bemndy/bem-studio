import logging

STEPS = 5


def run(notebook):
    """Fake stage: doubles a_data. Needs stage a to have run."""
    logging.info("Executing stage b")
    notebook.current_stage["status"] = "In Progress"

    # Implicit dependency: check our own inputs up front, before any work,
    # and fail with a message that says what we needed.
    if not notebook.data.get("a_data"):
        notebook.current_stage["status"] = "Failed"
        raise RuntimeError("stage b needs a_data from stage a")

    previous_data = notebook.data["a_data"]

    for step in range(STEPS):
        notebook.report((step + 1) / STEPS)

    notebook.data["b_data"] = [value * 2 for value in previous_data]
    notebook.current_stage["status"] = "Completed"