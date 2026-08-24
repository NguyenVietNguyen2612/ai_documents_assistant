import json
import os

from langsmith import Client
from dotenv import load_dotenv


load_dotenv()


def main():
    dataset_file = "./dataset.json"

    with open(dataset_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    client = Client()

    dataset_name = "CS338-RAG-Evaluation"

    # Create dataset
    dataset = client.create_dataset(
        dataset_name=dataset_name,
        description="Evaluation dataset for the CS338 RAG project."
    )

    # Add examples
    for item in data:
        client.create_example(
            inputs={
                "question": item["question"]
            },
            outputs={
                "answer": item["expected_answer"]
            },
            dataset_id=dataset.id
        )

    print(f"Dataset created successfully.")
    print(f"Dataset name: {dataset_name}")
    print(f"Number of examples: {len(data)}")


if __name__ == "__main__":
    main()