import logging

STEPS = 5


def run(notebook):
    """Fake stage: produces a_data. No prerequisites."""
    logging.info("Executing stage a")
    notebook.current_stage["status"] = "In Progress"

    for step in range(STEPS):
        # placeholder for real work
        notebook.report((step + 1) / STEPS)

    notebook.data["a_data"] = [1.3, 5.4, 6.7]
    notebook.current_stage["status"] = "Completed"