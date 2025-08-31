""" The app file which takes user input and passes it to the llm which would have access
to the retriever tools. These tools would be used by the llm for answering the questions.
"""

from tools import knowledge_tool, krithi_tool, raga_index_tool, multi_search
from semantic_layer import Prompt
from models import Models

def select_tools(user_question):
    """Intelligently select which tools to use based on the user's question"""
    question_lower = user_question.lower()
    selected_tools = []
    
    # Check for knowledge-related queries
    if any(keyword in question_lower for keyword in ["what is", "explain", "theory", "literature", "raga", "scale", "prayoga"]):
        selected_tools.append(("knowledge_tool", knowledge_tool))
    
    # Check for raga-specific queries
    if any(keyword in question_lower for keyword in ["raga", "melakarta", "janya", "alias", "list", "number"]):
        selected_tools.append(("raga_index_tool", raga_index_tool))
    
    # Check for composition-related queries
    if any(keyword in question_lower for keyword in ["krithi", "kriti", "composition", "lyrics", "composer", "tala", "song"]):
        selected_tools.append(("krithi_tool", krithi_tool))
    
    # If multiple categories or general query, use multi_search
    if len(selected_tools) > 1 or "carnatic music" in question_lower:
        selected_tools.append(("multi_search", multi_search))
    
    # If no specific tools selected, default to knowledge_tool
    if not selected_tools:
        selected_tools.append(("knowledge_tool", knowledge_tool))
    
    return selected_tools

def get_answer(user_question):
    """Get answer from the LLM using appropriate tools"""
    try:
        # Initialize models and semantic layer
        semantic_layer_obj = Prompt(user_question)
        llm_model_obj = Models()
        llm_model = llm_model_obj.getLLM()
        
        # Select appropriate tools based on user input
        selected_tools = select_tools(user_question)
        
        # Execute selected tools and collect results
        tool_results = []
        for tool_name, tool_func in selected_tools:
            try:
                if tool_name == "multi_search":
                    result = multi_search.invoke({
                        "query": user_question,
                        "categories": ["Literature", "Raga", "Krithis"],
                        "k_each": 4
                    })
                else:
                    result = tool_func.invoke(user_question)
                
                tool_results.append(f"Results from {tool_name}:\n{result}")
                
            except Exception as e:
                tool_results.append(f"Error with {tool_name}: {e}")
        
        # Get the prompt string ONLY from semantic layer
        semantic_prompt = semantic_layer_obj.getPromptStr()
        
        # Combine semantic prompt with tool results
        final_prompt = f"""{semantic_prompt}

Retrieved Information from Knowledge Base:
{chr(10).join(tool_results)}

Please provide a comprehensive answer based on the information above."""

        # Get response from the LLM
        response = llm_model.invoke(final_prompt)
        return response.content
        
    except Exception as e:
        return f"Error: {e}"

def main():
    """Main Q&A interface"""
    print("🎵 Carnatic Music Assistant 🎵")
    print("Ask me anything about Carnatic music theory, ragas, compositions, and more!")
    print("Type 'quit' or 'exit' to end the session.\n")
    
    while True:
        try:
            # Get user input
            user_input = input("\n❓ Your question: ").strip()
            
            # Check for exit command
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Thank you for using Carnatic Music Assistant! Goodbye!")
                break
            
            # Skip empty input
            if not user_input:
                continue
            
            print("\n🔍 Searching for information...")
            
            # Get answer
            answer = get_answer(user_input)
            
            print("\n💡 Answer:")
            print("-" * 50)
            print(answer)
            print("-" * 50)
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ An error occurred: {e}")

if __name__ == "__main__":
    main()