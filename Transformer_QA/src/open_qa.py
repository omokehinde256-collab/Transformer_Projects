from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from retriever import Retriever


class OpenDomainQA:

    def __init__(self):

        print("Loading generative model...")

        self.tokenizer = AutoTokenizer.from_pretrained(
            "google/flan-t5-small"
        )

        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            "google/flan-t5-small"
        )

        print("Generative model loaded!")

        self.retriever = Retriever()

        documents = [
            "The Federal University of Technology, Akure (FUTA) was established in 1981.",
            "FUTA is located in Akure, Ondo State, Nigeria.",
            "FUTA focuses on science, engineering, technology and related disciplines.",
            "The University of Lagos was established in 1962.",
            "The University of Ibadan was established in 1948.",
            "The University of Nigeria, Nsukka was established in 1960."
        ]

        self.retriever.add_documents(documents)

    def answer(self, question):

        results = self.retriever.search(
            question,
            top_k=3
        )

        context = "\n".join(
            result["document"]
            for result in results
        )

        prompt = f"""
Answer the question using only the information in the context.

Context:
{context}

Question:
{question}

Answer:
"""

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True
        )

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=80
        )

        answer = self.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        )

        return answer, results


if __name__ == "__main__":

    qa = OpenDomainQA()

    question = "When was FUTA established?"

    answer, results = qa.answer(question)

    print("\n==============================")
    print("OPEN-DOMAIN QA")
    print("==============================")

    print("Question:", question)
    print("Answer:", answer)

    print("\nRetrieved information:")

    for result in results:
        print("-", result["document"])

    print("==============================")