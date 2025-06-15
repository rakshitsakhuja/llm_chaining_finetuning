from typing import List
import asyncio
import json
from tqdm import tqdm
from llm_data_generator.generators.data_generator import DataGenerator

class DatasetManager:
    """Class for managing the dataset generation and storage"""
    def __init__(self, output_file: str = "synthetic_dataset.jsonl", batch_size: int = 3, delay_between_batches: int = 300):
        self.output_file = output_file
        self.batch_size = batch_size
        self.delay_between_batches = delay_between_batches  # 5 minutes default delay

    async def process_batch(self, topics: List[str], data_generator: DataGenerator) -> List[dict]:
        """Process a batch of topics"""
        print(f"\nDEBUG: Processing batch of {len(topics)} topics: {topics}")
        tasks = [data_generator.process_topic(topic) for topic in topics]
        results = await asyncio.gather(*tasks)
        
        # Count total Q&A pairs in this batch
        total_qa_pairs = sum(len(topic_results) for topic_results in results)
        print(f"DEBUG: Batch completed with {total_qa_pairs} total Q&A pairs")
        
        # Flatten results
        qa_pairs = [qa_pair for topic_results in results for qa_pair in topic_results]
        print(f"DEBUG: Flattened batch results into {len(qa_pairs)} Q&A pairs")
        return qa_pairs

    async def append_to_file(self, qa_pairs: List[dict]) -> None:
        """Append Q&A pairs to the output file"""
        print(f"DEBUG: Writing {len(qa_pairs)} Q&A pairs to file")
        with open(self.output_file, "a", encoding='utf-8') as f:
            for qa_pair in tqdm(qa_pairs, desc="Writing to file"):
                json.dump(qa_pair, f, ensure_ascii=False)
                f.write("\n")
        print(f"DEBUG: Successfully wrote to file")

    async def process_topics(self, topics: List[str], data_generator: DataGenerator) -> None:
        """Process all topics in batches with delays between batches"""
        print(f"DEBUG: Starting to process {len(topics)} total topics in batches of {self.batch_size}")
        
        # Clear the output file if it exists
        open(self.output_file, "w").close()
        print(f"DEBUG: Cleared output file {self.output_file}")
        
        total_qa_pairs = 0
        # Process topics in batches
        for i in tqdm(range(0, len(topics), self.batch_size)):
            batch_topics = topics[i:i + self.batch_size]
            print(f"\nDEBUG: Starting batch {i//self.batch_size + 1} of {(len(topics) + self.batch_size - 1)//self.batch_size}")
            print(f"DEBUG: Processing topics: {batch_topics}")
            
            # Process current batch
            qa_pairs = await self.process_batch(batch_topics, data_generator)
            total_qa_pairs += len(qa_pairs)
            print(f"DEBUG: Current batch produced {len(qa_pairs)} Q&A pairs")
            print(f"DEBUG: Total Q&A pairs so far: {total_qa_pairs}")
            
            # Append results to file
            await self.append_to_file(qa_pairs)
            
            # If this isn't the last batch, wait before processing the next batch
            if i + self.batch_size < len(topics):
                print(f"\nDEBUG: Waiting {self.delay_between_batches} seconds before next batch...")
                await asyncio.sleep(self.delay_between_batches)
        
        print(f"\nDEBUG: Processing complete. Total Q&A pairs generated: {total_qa_pairs}")
        print(f"DEBUG: Expected Q&A pairs (3 questions per topic): {len(topics) * 3}")


        