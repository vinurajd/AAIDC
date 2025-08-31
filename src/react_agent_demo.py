"""
Demo script for the React Agent system
Shows how the agent critiques and refines LLM responses
"""
from semantic_layer import ReactAgent
from models import Models

def demo_react_agent():
    """Demonstrate the React Agent workflow"""
    
    print("🎭 React Agent Demo - Carnatic Music Assistant")
    print("=" * 60)
    
    # Initialize models
    llm_model_obj = Models()
    llm_model = llm_model_obj.getLLM()
    
    # Initialize React Agent
    react_agent = ReactAgent(llm_model)
    
    # Sample user question
    user_question = "What is raga Mayamalavagowla?"
    
    # Sample tool results (simulated)
    tool_results = [
        "Results from knowledge_tool:\nRaga Mayamalavagowla is a fundamental raga in Carnatic music...",
        "Results from raga_index_tool:\nMayamalavagowla is the 15th melakarta raga..."
    ]
    
    # Sample conversation context
    conversation_context = "User: Tell me about Carnatic music\nAssistant: Carnatic music is a classical music tradition from South India..."
    
    # Sample initial response (simulated)
    initial_response = """Raga Mayamalavagowla is a raga in Carnatic music. It has some notes and is used in compositions. The raga has arohanam and avarohanam patterns."""
    
    print(f"🎯 User Question: {user_question}")
    print(f"📚 Tool Results: {len(tool_results)} sources retrieved")
    print(f"💬 Conversation Context: {len(conversation_context)} characters")
    print(f"📝 Initial Response: {len(initial_response)} characters")
    print("\n" + "=" * 60)
    
    # Stage 1: Critique
    print("🔍 Stage 1: Critiquing the initial response...")
    critique = react_agent.critique_response(
        user_question, initial_response, tool_results, conversation_context
    )
    print(f"📋 Critique:\n{critique}")
    print("\n" + "=" * 60)
    
    # Stage 2: Refine
    print("✨ Stage 2: Refining based on critique...")
    refined_response = react_agent.refine_response(
        user_question, initial_response, critique, tool_results, conversation_context
    )
    print(f"🎨 Refined Response:\n{refined_response}")
    print("\n" + "=" * 60)
    
    # Complete workflow
    print("🚀 Complete React Agent Workflow...")
    react_result = react_agent.process_with_react(
        user_question, initial_response, tool_results, conversation_context
    )
    
    print("✅ Final Result:")
    print(f"   - Original Length: {len(react_result['original_response'])} characters")
    print(f"   - Refined Length: {len(react_result['refined_response'])} characters")
    print(f"   - Improvement Applied: {react_result['improvement_applied']}")
    
    return react_result

if __name__ == "__main__":
    try:
        result = demo_react_agent()
        print("\n🎉 React Agent Demo completed successfully!")
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
