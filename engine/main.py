import pipeline
import logging

logging.basicConfig(level=logging.INFO)

def main():
    # TEST
    notebook = pipeline.Notebook(stages=["a", "b"], source="API", audio="./audio.wav")
    pipeline.run_pipeline(notebook)

if __name__ == "__main__":
    main()