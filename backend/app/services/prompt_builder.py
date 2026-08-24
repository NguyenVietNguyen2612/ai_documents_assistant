class PromptBuilder:

    def build(
        self,
        question: str,
        context: str,
    ) -> str:

        if not context or not context.strip():
            return f"""
You are a helpful AI document assistant.
Please respond directly to the user's conversational message.

User Message:
{question}

Answer:
"""

        return f"""
You are an AI assistant answering questions based strictly on the provided documents.

Please adhere strictly to the following rules:
1. CRITICAL: You MUST answer in the EXACT SAME LANGUAGE as the user's question. If the question is in English, answer in English, even if the provided context is in Vietnamese or another language.
2. Do not use asterisks (*) for formatting (no bolding, no bullet points). Use plain text formatting.
3. Be concise and direct. Keep your answer brief, ideally within 2-3 sentences.
4. Do NOT place citations in the middle of your sentences. You must gather all your source citations and list them only once at the VERY END of your entire response in this format: "Source: [Document Name], Page [Page Number]".
5. If the context contains mathematical formulas or equations, you may use LaTeX format (e.g. $...$ for inline or $$...$$ for block) to display them properly.

If the answer cannot be found in the context, say that the provided documents do not contain enough information to answer the question. Do not invent information.

Context:
--------------------
{context}
--------------------

Question:
{question}

Answer:
"""