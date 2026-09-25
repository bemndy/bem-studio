import logging
import time

def run(Notebook):
    logging.info("Executing hook b")
    Notebook.current_stage["status"] = "In Progress"

    previous_data = Notebook.data.get("a_data", [])
    for i in range(5):
        time.sleep(1)  # Simulate some work being done
        Notebook.current_stage["progress"] += 1 / 5  # Update progress
        logging.info(f"Hook b progress: {Notebook.current_stage['progress'] * 100:.2f}%")
        if not Notebook.data.get("a_data"):
            logging.error("No data from previous stage 'a' found. Terminating from stage 'b'.")
            Notebook.current_stage["status"] = "Failed"
            raise RuntimeError("No data from previous stage 'a' found. Terminating from stage 'b'.")

        Notebook.data["b_data"] = list(map(lambda x: x * 2, previous_data))  # Update the data with the processed values
        Notebook.current_stage["status"] = "Completed"

def b_dummy(Notebook):
    print("hi")