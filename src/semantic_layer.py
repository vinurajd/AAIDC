

from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.schema import SystemMessage
from langchain.memory import ConversationBufferMemory
import time

class Prompt:
    def __init__(self, user_question):
        self.user_query = user_question
        
    def getSystemMessage(self):
        """Get the system message for the Carnatic music agent"""
        system_content = """You are an expert assistant on Carnatic music. You have access to several tools to help answer questions:

1. knowledge_tool: Use this for questions about Carnatic music theory, literature, ragas, scales, and prayogas
2. raga_index_tool: Use this for looking up raga information, aliases, and melakarta mappings
3. krithi_tool: Use this for searching compositions, lyrics, composers, and explanations
4. multi_search: Use this for complex queries that span multiple categories

When answering questions:
- Always use the appropriate tools to retrieve relevant information
- Provide comprehensive answers based on the retrieved content
- Cite your sources when possible
- If a question spans multiple areas, use multiple tools as needed
- Consider conversation context to provide more relevant and contextual responses

Current question: {user_query}"""
        
        return SystemMessage(content=system_content)
    
    def getPromptTemplate(self):
        """Get the ChatPromptTemplate for the agent"""
        system_message = self.getSystemMessage()
        
        prompt = ChatPromptTemplate.from_messages([
            system_message,
            ("human", "{input}"),
            ("assistant", "{agent_scratchpad}"),
        ])
        
        return prompt
    
    def getPromptStr(self):
        """Get a simple string prompt (keeping for backward compatibility)"""
        template_str = """ You are an expert assistant.
        Answer the following question concisely:
        Question: {user_query}
        """
        
        prompt_str = PromptTemplate(
            input_variables=["user_query"],
            template=template_str
        )
        final_prompt = prompt_str.format(user_query=self.user_query)
        return final_prompt

class ReactAgent:
    """React Agent that critiques and refines LLM outputs for better quality"""
    
    def __init__(self, llm_model):
        self.llm_model = llm_model
        self.critic_prompt = self._create_critic_prompt()
        self.refiner_prompt = self._create_refiner_prompt()
    
    def _create_critic_prompt(self):
        """Create the prompt for the critic LLM"""
        return """You are an expert critic and evaluator specializing in Carnatic music knowledge. Your role is to critically evaluate the quality, accuracy, and completeness of responses about Carnatic music.

EVALUATION CRITERIA:
1. **Accuracy**: Is the information factually correct?
2. **Completeness**: Does it answer the question fully?
3. **Relevance**: Is the content relevant to the user's question?
4. **Clarity**: Is the explanation clear and understandable?
5. **Depth**: Does it provide sufficient detail and context?
6. **Source Integration**: Are the retrieved sources properly utilized?
7. **Conversation Context**: Does it consider previous conversation context?

EVALUATION FORMAT:
Provide your evaluation in the following structure:

SCORE: [1-10] (10 being excellent)
STRENGTHS: [List key strengths]
WEAKNESSES: [List areas for improvement]
CRITIQUE: [Detailed analysis]
SUGGESTIONS: [Specific recommendations for improvement]

Be constructive but thorough in your evaluation."""

    def _create_refiner_prompt(self):
        """Create the prompt for the refiner LLM"""
        return """You are an expert Carnatic music content refiner. Your role is to take the original response and the critic's feedback to create an improved, refined version.

REFINEMENT GUIDELINES:
1. **Address Criticisms**: Fix all identified weaknesses and issues
2. **Enhance Strengths**: Build upon what was done well
3. **Improve Clarity**: Make explanations clearer and more accessible
4. **Add Depth**: Include more relevant details where needed
5. **Better Structure**: Organize information logically
6. **Source Integration**: Better utilize and cite retrieved information
7. **Context Awareness**: Maintain conversation flow and context

Create a refined response that is significantly better than the original while maintaining the core information and addressing all feedback points."""

    def critique_response(self, user_question: str, initial_response: str, tool_results: list, conversation_context: str):
        """Critique the initial LLM response"""
        critique_prompt = f"""{self.critic_prompt}

USER QUESTION: {user_question}

CONVERSATION CONTEXT:
{conversation_context}

RETRIEVED INFORMATION:
{chr(10).join(tool_results)}

INITIAL RESPONSE TO EVALUATE:
{initial_response}

Please provide your critical evaluation of this response."""

        try:
            critique = self.llm_model.invoke(critique_prompt)
            return critique.content
        except Exception as e:
            return f"Critique failed: {e}"

    def refine_response(self, user_question: str, initial_response: str, critique: str, tool_results: list, conversation_context: str):
        """Refine the response based on the critique"""
        refinement_prompt = f"""{self.refiner_prompt}

USER QUESTION: {user_question}

CONVERSATION CONTEXT:
{conversation_context}

RETRIEVED INFORMATION:
{chr(10).join(tool_results)}

ORIGINAL RESPONSE:
{initial_response}

CRITIC'S FEEDBACK:
{critique}

Please create a refined, improved response based on the feedback."""

        try:
            refined_response = self.llm_model.invoke(refinement_prompt)
            return refined_response.content
        except Exception as e:
            return f"Refinement failed: {e}. Using original response: {initial_response}"

    def process_with_react(self, user_question: str, initial_response: str, tool_results: list, conversation_context: str):
        """Complete React Agent workflow: critique -> refine -> return improved response"""
        
        # Stage 1: Critique the initial response
        print("🎭 Stage 1: Critiquing initial response...")
        critique = self.critique_response(user_question, initial_response, tool_results, conversation_context)
        
        # Stage 2: Refine based on critique
        print("✨ Stage 2: Refining response based on critique...")
        refined_response = self.refine_response(user_question, initial_response, critique, tool_results, conversation_context)
        
        return {
            "original_response": initial_response,
            "critique": critique,
            "refined_response": refined_response,
            "improvement_applied": True
        }

class ConversationManager:
    """Manages conversation memory and context for the Carnatic Music Assistant"""
    
    def __init__(self):
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        self.messages = []
    
    def add_message(self, role: str, content: str, tools_used=None, react_details=None):
        """Add a message to the conversation with optional React Agent details"""
        timestamp = time.strftime("%H:%M")
        message = {
            "role": role,
            "content": content,
            "timestamp": timestamp,
            "tools_used": tools_used,
            "react_details": react_details  # Store React Agent details
        }
        self.messages.append(message)
        return message
    
    def get_conversation_context(self, max_messages: int = 4):
        """Get conversation context for the LLM"""
        chat_history = self.memory.chat_memory.messages
        
        if not chat_history:
            return "No previous conversation."
        
        # Get the last N messages for context
        recent_messages = chat_history[-max_messages:]
        context_lines = []
        
        for msg in recent_messages:
            role = "User" if msg.type == "human" else "Assistant"
            context_lines.append(f"{role}: {msg.content}")
        
        return chr(10).join(context_lines)
    
    def create_context_aware_prompt(self, base_prompt: str, user_question: str, tool_results: list):
        """Create a context-aware prompt with conversation history"""
        conversation_context = self.get_conversation_context()
        
        context_prompt = f"""{base_prompt}

Previous Conversation Context:
{conversation_context}

Current Question: {user_question}

Retrieved Information from Knowledge Base:
{chr(10).join(tool_results)}

Please provide a comprehensive answer based on the information above. Consider the conversation context to provide more relevant and contextual responses."""
        
        return context_prompt
    
    def save_to_memory(self, user_question: str, ai_response: str):
        """Save the conversation to LangChain memory"""
        self.memory.chat_memory.add_user_message(user_question)
        self.memory.chat_memory.add_ai_message(ai_response)
    
    def clear_conversation(self):
        """Clear both conversation memory and chat messages"""
        self.memory.clear()
        self.messages = []
    
    def get_memory_stats(self):
        """Get statistics about the conversation memory"""
        return {
            "total_messages": len(self.messages),
            "memory_messages": len(self.memory.chat_memory.messages)
        }
# # Example usage
# user_input = "Explain the difference between supervised and unsupervised learning."

# print(final_prompt)
