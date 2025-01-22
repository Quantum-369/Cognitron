import os
import asyncio
import re
from datetime import datetime, timezone
import google.generativeai as genai
from openai import OpenAI
from pinecone import Pinecone
from typing import List, Dict, Optional

# Initialize clients
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("cognitron")

# Configure Gemini
genai.configure(api_key=os.environ["GEMINI_API_KEY"])

# Create the model
generation_config = {
    "temperature": 0.7,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 8192,
}

model = genai.GenerativeModel(
    model_name="gemini-2.0-flash-exp",
    generation_config=generation_config,
    system_instruction="""You are an intelligent assistant with access to the user's past memories and conversations. 
When responding:
1. Use the provided memory context naturally in your responses
2. If no relevant memories are found, respond based on the current conversation
3. Always maintain a friendly and contextual conversation flow
4. Be specific about what information you found in memories
5. For personal information queries, always check memories first"""
)

# Memory command patterns
MEMORY_COMMAND_PATTERNS = {
    "remember": r"^remember this: (.+)",
    "delete": r"^delete: (.+)"
}

def detect_memory_command(user_input: str) -> tuple:
    """Simple regex-based command detection"""
    user_input = user_input.strip().lower()
    for cmd, pattern in MEMORY_COMMAND_PATTERNS.items():
        match = re.match(pattern, user_input, re.IGNORECASE)
        if match:
            return cmd, match.group(1).strip()
    return None, None

async def handle_memory_command(command: str, text: str):
    """Process explicit memory commands"""
    if command == "remember":
        interaction = {
            "user_input": f"[EXPLICIT] {text}",
            "agent_response": "User-commanded memory storage",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "memory_type": "USER_DIRECTIVE"
        }
        await store_memory(interaction)
    elif command == "delete":
        try:
            # Generate embedding for deletion query
            embedding_response = openai_client.embeddings.create(
                model="text-embedding-3-small",
                input=text
            )
            embedding = embedding_response.data[0].embedding
            
            # Find similar memories with a high similarity threshold
            results = index.query(
                vector=embedding,
                top_k=5,
                include_metadata=True,
                filter={"memory_type": {"$ne": "USER_DIRECTIVE"}}  # Exclude explicit commands
            )
            
            # Delete only high-confidence matches
            to_delete = [m.id for m in results.matches if m.score > 0.6]  # High threshold
            if to_delete:
                index.delete(ids=to_delete)
                return len(to_delete)
            return 0
        except Exception as e:
            print(f"Deletion error: {str(e)}")
            return 0

def classify_memory(user_input: str, agent_response: str) -> Optional[str]:
    """Classify the interaction for memory storage"""
    classification_prompt = """You are tasked with classifying user input into specific memory types **only if the information is personal or user-related**. Ignore general knowledge or facts that the LLM already possesses and do not classify or store them. Only classify inputs that provide unique information about the user or their experiences.

The memory types are as follows:

LONG_TERM_MEMORY:
- **EPISODIC**: Personal experiences or events shared by the user.
- **DECLARATIVE**: Explicit factual information about the user.
- **PROCEDURAL**: Instructions, processes, or how-to information related to the user's input.

SPECIAL_MEMORY:
- **SPATIAL**: Location-specific information shared by the user.
- **AUTOBIOGRAPHICAL**: User preferences, traits, or life details.
- **FLASHBULB**: Highly significant or emotional events in the user's life.
- **EMOTIONAL**: Sentiment and emotional content expressed by the user.

### Rules:
1. **DO NOT classify or store general knowledge or facts (e.g., "The Earth orbits the Sun" or "The capital of France is Paris").** Ignore such inputs.
2. Analyze the user input and classify **only if it provides unique, user-related information**.
3. Respond with ONLY the memory type in UPPERCASE (e.g., "EPISODIC", "DECLARATIVE").
4. If the input is irrelevant or contains general knowledge, do not classify it and respond with "IGNORE".

### Examples:

**Good Examples:**
- "My birthday is January 15th" → **DECLARATIVE**
- "I feel really happy today" → **EMOTIONAL**
- "I live in New York" → **SPATIAL**
- "Let me explain how I bake cookies" → **PROCEDURAL**
- "I remember when I graduated from college" → **EPISODIC**
- "I love Italian food, especially pasta" → **AUTOBIOGRAPHICAL**
- "Last year, I got married, and it was the happiest day of my life" → **FLASHBULB**

**Bad Examples (Ignored):**
- "The Earth orbits the Sun" → **IGNORE**
- "The Eiffel Tower is in Paris" → **IGNORE**
- "Water boils at 100 degrees Celsius" → **IGNORE**

**Input Classification Workflow:**
1. Identify if the input is user-related (personal experience, sentiment, preferences, or location).
2. If yes, classify it into the correct memory type.
3. If no, respond with "IGNORE".

User Input: {input}
Agent Response: {response}
"""
    
    try:
        response = model.generate_content(classification_prompt.format(
            input=user_input, 
            response=agent_response
        ))
        return response.text.strip().upper()
    except Exception as e:
        print(f"Classification error: {str(e)}")
        return None

async def store_memory(interaction: Dict):
    """Store the interaction in vector database"""
    try:
        memory_text = f"User: {interaction['user_input']} | Agent: {interaction['agent_response']}"
        
        # Generate embedding using OpenAI
        embedding_response = openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=memory_text
        )
        embedding = embedding_response.data[0].embedding

        # Store in Pinecone
        index.upsert(
            vectors=[(
                interaction["timestamp"],
                embedding,
                interaction
            )]
        )
        print(f"Memory stored: {interaction['timestamp']}")
    except Exception as e:
        print(f"Error storing memory: {str(e)}")

def get_relevant_memories(query: str, top_k: int = 3) -> List[Dict]:
    """Retrieve relevant memories using semantic search"""
    try:
        query_embedding = openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=query
        ).data[0].embedding

        results = index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )

        memories = []
        for match in results.matches:
            if match.score > 0.2:
                memory = match.metadata
                memory["similarity_score"] = match.score
                memories.append(memory)
        
        return memories
    except Exception as e:
        print(f"Error retrieving memories: {str(e)}")
        return []

def format_memories_for_context(memories: List[Dict]) -> str:
    """Format retrieved memories into a context string"""
    if not memories:
        return "No relevant memories found."
    
    context = "Relevant memories:\n"
    for idx, memory in enumerate(memories, 1):
        context += f"{idx}. Previous interaction:\n"
        context += f"   User: {memory['user_input']}\n"
        context += f"   Response: {memory['agent_response']}\n"
        context += f"   (Memory type: {memory['memory_type']})\n\n"
    return context

def initialize_chat_model():
    """Initialize and return the chat model"""
    model = genai.GenerativeModel(
        model_name="gemini-2.0-flash-exp",
        generation_config=generation_config,
        system_instruction="""You are an intelligent assistant with access to the user's past memories and conversations. 
When responding:
1. Use the provided memory context naturally in your responses
2. If no relevant memories are found, respond based on the current conversation
3. Always maintain a friendly and contextual conversation flow
4. Be specific about what information you found in memories
5. For personal information queries, always check memories first"""
    )
    return model

async def process_message_with_memory(model, user_input: str) -> str:
    """Process a message with memory storage and retrieval"""
    try:
        # First check for memory commands
        command, memory_text = detect_memory_command(user_input)
        if command:
            if command == "delete":
                deleted_count = await handle_memory_command(command, memory_text)
                if deleted_count > 0:
                    return f"Deleted {deleted_count} related memories."
                return "No matching memories found to delete."
            else:
                await handle_memory_command(command, memory_text)
                return "I've specifically memorized this information."

        # Retrieve relevant memories
        memories = get_relevant_memories(user_input)
        memory_context = format_memories_for_context(memories)
        
        # Prepare enhanced prompt with memory context
        enhanced_prompt = f"""Memory Context: {memory_context if memories else "No relevant memories found."}

User Question: {user_input}

Instructions:
1. Only reference memories if they are DIRECTLY relevant to the user's question.
2. If no relevant memories exist, respond based on the current conversation only.
3. Do not mention memories unless they are explicitly needed to answer the question.
4. Keep the response concise and avoid unnecessary details.

Please provide your response:"""

        # Get response from Gemini
        chat = model.start_chat(history=[])
        response = chat.send_message(enhanced_prompt)
        
        # Extract response text
        response_text = ""
        for part in response.parts:
            if hasattr(part, 'text'):
                response_text += part.text
        
        response_text = response_text.strip()

        # Store the interaction asynchronously
        memory_type = classify_memory(user_input, response_text)
        if memory_type and memory_type != "IGNORE":
            interaction = {
                "user_input": user_input,
                "agent_response": response_text,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "memory_type": memory_type
            }
            # Create a task for memory storage
            asyncio.create_task(store_memory(interaction))
        
        return response_text

    except Exception as e:
        print(f"Error processing message: {str(e)}")
        return f"I apologize, but I encountered an error while processing your message: {str(e)}"

async def chat_loop():
    """Main chat loop with integrated memory storage and retrieval"""
    print("Chat started. Type 'exit' to end the conversation.")
    chat = model.start_chat(history=[])
    
    while True:
        user_input = input("\nYou: ").strip()
        
        if user_input.lower() == 'exit':
            print("Chat ended. Goodbye!")
            break
            
        try:
            # Handle memory commands first
            command, memory_text = detect_memory_command(user_input)
            if command:
                if command == "delete":
                    deleted_count = await handle_memory_command(command, memory_text)
                    if deleted_count > 0:
                        print(f"\nAgent: Deleted {deleted_count} related memories.")
                    else:
                        print("\nAgent: No matching memories found to delete.")
                    continue  # Skip memory storage for deletion commands
                else:
                    await handle_memory_command(command, memory_text)
                    print("\nAgent: I've specifically memorized this information.")
                    continue  # Skip memory storage for explicit "remember" commands
                
            # Retrieve relevant memories
            memories = get_relevant_memories(user_input)
            memory_context = format_memories_for_context(memories)
            
            # Prepare enhanced prompt with memory context
            enhanced_prompt = f"""Memory Context: {memory_context if memories else "No relevant memories found."}

User Question: {user_input}

Instructions:
1. Only reference memories if they are DIRECTLY relevant to the user's question.
2. If no relevant memories exist, respond based on the current conversation only.
3. Do not mention memories unless they are explicitly needed to answer the question.
4. Keep the response concise and avoid unnecessary details.

Please provide your response:"""

            # Get response from Gemini
            response = chat.send_message(enhanced_prompt)
            
            # Extract response text
            response_text = ""
            for part in response.parts:
                if hasattr(part, 'text'):
                    response_text += part.text
            
            response_text = response_text.strip()
            print("\nAgent:", response_text)

            # Classify and store the interaction (skip for commands)
            if not command:  # Only store if it's not a command
                memory_type = classify_memory(user_input, response_text)
                if memory_type and memory_type != "IGNORE":
                    interaction = {
                        "user_input": user_input,
                        "agent_response": response_text,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "memory_type": memory_type
                    }
                    await store_memory(interaction)
            
        except Exception as e:
            print(f"\nAn error occurred: {str(e)}")
            print("Please try again.")

if __name__ == "__main__":
    asyncio.run(chat_loop())