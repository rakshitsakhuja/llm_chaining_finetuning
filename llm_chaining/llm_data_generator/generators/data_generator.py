from llm_data_generator.clients.llm_client import LLMClient
import json
import re

class DataGenerator:
    """Class for generating synthetic training data"""
    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client
    
    def extract_questions(self, sub_questions: str) -> list:
        """Extract individual questions from raw sub_question text"""
        lines = sub_questions.strip().split('\n')
        questions = []
        for line in lines:
            if line.strip() and (line.strip().startswith("1.") or line.strip().startswith("2.") or line.strip().startswith("3.")):
                questions.append(line.split('.', 1)[1].strip())
        print(f"DEBUG: Extracted {len(questions)} questions from text")
        return questions

    async def generate_sub_questions(self, topic: str) -> str:
        """Generate sub-questions for a given topic"""
        prompt = f"""
        You are a domain expert in Machine Learning and Deep Learning.

        Given the technical topic: '{topic}', generate 3 highly specific, technical sub-questions that would help someone deeply understand the internal workings of this topic.

        Focus on internal mechanisms, algorithms, computations, and edge cases.
        """
        questions = await self.llm_client.call_openai(prompt)
        print(f"DEBUG: Generated questions for topic '{topic}':\n{questions}")
        return questions

    # async def generate_answers(self, sub_questions: str) -> str:
    #     """Generate answers for the sub-questions"""
    #     prompt = f"""
    #     You are an advanced ML research scientist.

    #     For each of the following questions:

    #     {sub_questions}

    #     Write extremely detailed, technically accurate answers with equations(if required), terminology, examples, and practical intuition where appropriate.

    #     Avoid hallucination. Only use verified knowledge in ML/DL literature.
    #     """
        return await self.llm_client.call_claude(prompt)
    async def generate_answers(self, sub_questions: str) -> list:
        """Generate answers for each question individually"""
        questions = self.extract_questions(sub_questions)
        answers = []
        
        print(f"DEBUG: Starting to generate answers for {len(questions)} questions")
        for i, q in enumerate(questions, 1):
            print(f"DEBUG: Generating answer {i}/{len(questions)} for question: {q[:100]}...")
            prompt = f"""
            You are an advanced ML research scientist.

            Write a highly detailed, technically accurate answer to the following question:
            {q}

            Include equations if necessary, terminology, and practical intuition.
            Avoid hallucinations.
            """
            answer = await self.llm_client.call_claude(prompt)
            answers.append({"instruction": q.strip(), "response": answer.strip()})
            print(f"DEBUG: Completed answer {i}/{len(questions)}")
        
        print(f"DEBUG: Generated {len(answers)} answers")
        return answers

    async def process_topic(self, topic: str) -> list:
        """Process a single topic through the entire pipeline"""
        print(f"\nDEBUG: Starting to process topic: {topic}")
        sub_questions = await self.generate_sub_questions(topic)
        answers = await self.generate_answers(sub_questions)
        print(f"DEBUG: Completed processing topic '{topic}' with {len(answers)} Q&A pairs")
        return answers 