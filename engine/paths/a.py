import logging
import time

def a(Notebook):
    logging.info("Executing hook a")
    for i in range(5):
        time.sleep(1)  # Simulate some work being done
        Notebook.current_stage["progress"] += (i + 1) / 5  # Update progress
        logging.info(f"Hook a progress: f{current_stage['progress'] * 100:.2f}%")
    Notebook.data["a_data"] = [1.3, 5.4, 6.7]

def a_dummy(Notebook):
    print("hi")