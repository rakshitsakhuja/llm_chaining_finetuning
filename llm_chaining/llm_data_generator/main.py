import asyncio
from llm_data_generator.config.config import LLMConfig
from llm_data_generator.clients.llm_client import LLMClient
from llm_data_generator.generators.data_generator import DataGenerator
from llm_data_generator.managers.dataset_manager import DatasetManager
from llm_data_generator.config.constants import TOPICS

def main():
    # Initialize configuration
    config = LLMConfig()
    
    # Initialize clients and managers
    llm_client = LLMClient(config)
    data_generator = DataGenerator(llm_client)
    
    # Initialize dataset manager with batch processing parameters
    # Process 3 topics at a time with 5 minutes (300 seconds) delay between batches
    dataset_manager = DatasetManager(
        output_file="synthetic_dataset.jsonl",
        batch_size=3,
        delay_between_batches=60
    )

    # Run the processing pipeline
    asyncio.run(dataset_manager.process_topics(TOPICS, data_generator))

if __name__ == "__main__":
    main() 