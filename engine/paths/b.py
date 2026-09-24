import logging
import time

def b(Notebook):
    logging.info("Executing hook b")
    previous_data = Notebook.data.get("a_data", [])
    for i in range(5):
        time.sleep(1)  # Simulate some work being done
        Notebook.current_stage["progress"] += (i + 1) / 5  # Update progress
        logging.info(f"Hook b progress: f{Notebook.current_stage['progress'] * 100:.2f}%")
        Notebook.data["b_data"] = list(map(lambda x: x * 2, previous_data))  # Update the data with the processed values

def b_dummy(Notebook):
    print("hi")