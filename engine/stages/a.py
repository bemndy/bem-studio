import logging
import time

def run(Notebook):
    logging.info("Executing hook a")
    Notebook.current_stage["status"] = "In Progress"

    for i in range(5):
        time.sleep(1)  # Simulate some work being done
        Notebook.current_stage["progress"] += 1 / 5  # Update progress
        logging.info(f"Hook a progress: {Notebook.current_stage['progress'] * 100:.2f}%")
    Notebook.data["a_data"] = [1.3, 5.4, 6.7]
    Notebook.current_stage["status"] = "Completed"

def a_dummy(Notebook):
    print("hi")