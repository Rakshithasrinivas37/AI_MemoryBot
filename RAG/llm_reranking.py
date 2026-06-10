def llm_rerank(query, memories, client):
    if not memories:
        return []
    if len(memories) == 1:
        return memories

    memories_text = "\n".join([
        f"Memory {i+1}: {m}"
        for i, m in enumerate(memories)
    ])

    prompt = f"""You are ranking memories to answer a question.

        Question: "{query}"

        Memories:
        {memories_text}

        Which memory BEST answers the question?
        Rank ALL memories from most to least relevant.

        Rules:
        - If question asks about age → rank memories mentioning
        age numbers or years old FIRST
        - If question asks about name → rank memories with
        name information FIRST
        - Return ONLY numbers comma separated
        - Most relevant first
        - Example: 2,1 means Memory 2 is most relevant

        Your ranking:"""

    response = client.chat.completions.create(
        model      = "llama-3.3-70b-versatile",
        messages   = [{"role": "user", "content": prompt}],
        max_tokens = 20
    )

    try:
        raw     = response.choices[0].message.content.strip()
        print(f"  LLM ranking response: {raw}")

        # ✅ Extract only digits
        indices = [
            int(x.strip()) - 1
            for x in raw.replace(" ", "").split(",")
            if x.strip().isdigit()
        ]

        reranked = [
            memories[i]
            for i in indices
            if i < len(memories)
        ]

        # Add missed memories at end
        for m in memories:
            if m not in reranked:
                reranked.append(m)

        return reranked

    except Exception as e:
        print(f"⚠️ Rerank parse failed: {e}")
        return memories