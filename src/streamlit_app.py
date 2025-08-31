"""
Streamlit App for Carnatic Music Assistant
A beautiful chat-style interface for asking questions about Carnatic music
"""
import streamlit as st
from tools import knowledge_tool, krithi_tool, raga_index_tool, multi_search
from semantic_layer import Prompt, ConversationManager, ReactAgent
from models import Models
import time

# Page configuration
st.set_page_config(
    page_title="🎵 Carnatic Music Assistant",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize conversation manager in session state
if "conversation_manager" not in st.session_state:
    st.session_state.conversation_manager = ConversationManager()

# Initialize last_input tracking for Enter key support
if "last_input" not in st.session_state:
    st.session_state.last_input = ""

# Custom CSS for chat-style interface
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 1rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .sub-header {
        font-size: 1.2rem;
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .chat-container {
        background: #f8f9fa;
        border-radius: 15px;
        padding: 1rem;
        margin: 1rem 0;
        min-height: 400px;
        max-height: 600px;
        overflow-y: auto;
        border: 2px solid #e9ecef;
    }
    .chat-bubble {
        margin: 1rem 0;
        display: flex;
        flex-direction: column;
    }
    .user-bubble {
        align-items: flex-end;
    }
    .assistant-bubble {
        align-items: flex-start;
    }
    .bubble-content {
        max-width: 80%;
        padding: 1rem;
        border-radius: 20px;
        word-wrap: break-word;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    .user-content {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-bottom-right-radius: 5px;
    }
    .assistant-content {
        background: #e9ecef;
        color: #333;
        border-bottom-left-radius: 5px;
    }
    .bubble-time {
        font-size: 0.8rem;
        color: #666;
        margin: 0.5rem 0;
        opacity: 0.7;
    }
    .input-container {
        background: white;
        border-radius: 25px;
        padding: 1rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        margin: 1rem 0;
        border: 2px solid #e9ecef;
    }
    .stTextInput > div > div > input {
        border: none;
        outline: none;
        font-size: 1.1rem;
    }
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.5rem 1.5rem;
        font-weight: bold;
        transition: all 0.3s ease;
        margin-left: 1rem;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
    .tool-info {
        background: #f8f9fa;
        padding: 0.5rem;
        border-radius: 8px;
        border-left: 4px solid #28a745;
        margin: 0.5rem 0;
        font-size: 0.9rem;
    }
    .example-chips {
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem;
        margin: 1rem 0;
    }
    .example-chip {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        cursor: pointer;
        transition: all 0.3s ease;
        font-size: 0.9rem;
    }
    .example-chip:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.2);
    }
    .typing-indicator {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        color: #666;
        font-style: italic;
    }
    .dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: #667eea;
        animation: typing 1.4s infinite ease-in-out;
    }
    .dot:nth-child(2) { animation-delay: 0.2s; }
    .dot:nth-child(2) { animation-delay: 0.4s; }
    
    @keyframes typing {
        0%, 60%, 100% { transform: translateY(0); }
        30% { transform: translateY(-10px); }
    }
</style>
""", unsafe_allow_html=True)

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

def get_answer(user_question, use_react_agent=True):
    """Get answer from the LLM using appropriate tools, conversation memory, and optional React Agent refinement"""
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
        
        # Use conversation manager to create context-aware prompt
        conversation_manager = st.session_state.conversation_manager
        conversation_context = conversation_manager.get_conversation_context()
        context_prompt = conversation_manager.create_context_aware_prompt(
            semantic_prompt, user_question, tool_results
        )

        if use_react_agent:
            # Initialize React Agent
            react_agent = ReactAgent(llm_model)
            
            # Stage 1: Get initial response from LLM
            print("🚀 Stage 1: Generating initial response...")
            initial_response = llm_model.invoke(context_prompt)
            initial_content = initial_response.content
            
            # Stage 2: Use React Agent to critique and refine
            print("🎭 Stage 2: React Agent processing...")
            react_result = react_agent.process_with_react(
                user_question, 
                initial_content, 
                tool_results, 
                conversation_context
            )
            
            final_answer = react_result["refined_response"]
            react_details = react_result
            
        else:
            # Direct response without React Agent
            print("🚀 Generating direct response...")
            response = llm_model.invoke(context_prompt)
            final_answer = response.content
            react_details = None
        
        # Save conversation to memory using conversation manager
        conversation_manager.save_to_memory(user_question, final_answer)
        
        return final_answer, selected_tools, react_details
        
    except Exception as e:
        return f"Error: {e}", [], None

def clear_conversation():
    """Clear the conversation using the conversation manager"""
    try:
        # Clear conversation manager
        st.session_state.conversation_manager.clear_conversation()
        
        # Clear any confirmation states
        if "show_clear_confirm" in st.session_state:
            del st.session_state.show_clear_confirm
        if "show_clear_confirm_main" in st.session_state:
            del st.session_state.show_clear_confirm_main
        
        # Success message
        st.success("🗑️ Conversation history cleared successfully!")
        
        # Small delay to show the message
        time.sleep(1)
        
    except Exception as e:
        st.error(f"❌ Error clearing conversation: {e}")

def main():
    """Main Streamlit chat interface"""

    # Header
    st.markdown('<h1 class="main-header">🎵 Carnatic Music Assistant</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Chat with me about Carnatic music theory, ragas, compositions, and more!</p>', unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.header("🎯 About This Assistant")
        st.markdown("""
        This AI assistant specializes in Carnatic music and can help you with:

        🎼 **Music Theory** - Scales, ragas, prayogas  
        🎵 **Raga Information** - Melakarta, janya, aliases  
        📚 **Compositions** - Krithis, lyrics, composers  
        🔍 **Comprehensive Search** - Multi-category queries
        """)

        st.header("🛠️ Available Tools")
        st.markdown("""
        - **Knowledge Tool**: Theory & literature  
        - **Raga Index Tool**: Raga information  
        - **Krithi Tool**: Compositions  
        - **Multi Search**: Cross-category search
        """)

        st.header("💡 Tips")
        st.markdown("""
        - Be specific in your questions  
        - Ask about specific ragas, composers, or concepts  
        - Use natural language  
        - The assistant remembers previous questions for context
        """)

        # Conversation management
        st.header("🗂️ Conversation")

        if st.button("🗑️ Clear History", use_container_width=True, type="secondary"):
            if st.session_state.get("show_clear_confirm", False):
                clear_conversation()
                st.session_state.show_clear_confirm = False
                st.rerun()
            else:
                st.session_state.show_clear_confirm = True
                st.warning("⚠️ Are you sure? This will clear all conversation history.")
                if st.button("✅ Yes, Clear Everything", use_container_width=True, type="primary"):
                    clear_conversation()
                    st.session_state.show_clear_confirm = False
                    st.rerun()
                if st.button("❌ Cancel", use_container_width=True):
                    st.session_state.show_clear_confirm = False
                    st.rerun()

        # React Agent toggle
        st.header("🎭 React Agent")
        use_react_agent = st.checkbox("Enable React Agent", value=True)
        if use_react_agent:
            st.info("🎭 React Agent is enabled - responses will be critiqued and refined")
        else:
            st.warning("⚠️ React Agent is disabled - using direct responses")

        # Conversation stats
        conversation_manager = st.session_state.conversation_manager
        memory_stats = conversation_manager.get_memory_stats()
        st.markdown(f"**Total Messages**: {memory_stats['total_messages']}")
        st.markdown(f"**Memory Messages**: {memory_stats['memory_messages']}")

    # Main chat area
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:

        # Chat history container
        #st.markdown('<div class="chat-container">', unsafe_allow_html=True)

        conversation_manager = st.session_state.conversation_manager
        for message in conversation_manager.messages:
            if message["role"] == "user":
                st.markdown(f'''
                <div class="chat-bubble user-bubble">
                    <div class="bubble-content user-content">
                        {message["content"]}
                    </div>
                    <div class="bubble-time">{message["timestamp"]}</div>
                </div>
                ''', unsafe_allow_html=True)
            else:
                st.markdown(f'''
                <div class="chat-bubble assistant-bubble">
                    <div class="bubble-content assistant-content">
                        {message["content"]}
                    </div>
                    <div class="bubble-time">{message["timestamp"]}</div>
                </div>
                ''', unsafe_allow_html=True)

                if message.get("tools_used"):
                    st.markdown("**🛠️ Tools Used:**")
                    for tool_name, _ in message["tools_used"]:
                        st.markdown(f'<div class="tool-info">✅ {tool_name}</div>', unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

        # Input container (no big box on top)
        st.markdown('<div class="input-container">', unsafe_allow_html=True)

        col_input, col_send = st.columns([4, 1])
        with col_input:
            user_input = st.text_input(
                "Type your message...",
                placeholder="Ask me about Carnatic music...",
                key="chat_input",
                label_visibility="collapsed"
            )

        with col_send:
            send_button = st.button("🚀", use_container_width=True)

        st.markdown('</div>', unsafe_allow_html=True)

        # Handle input
        if (send_button or (user_input and user_input.strip() and user_input != st.session_state.get("last_input", ""))):
            st.session_state.last_input = user_input
            if user_input.strip():
                conversation_manager.add_message("user", user_input.strip())
                with st.spinner(""):
                    st.markdown('<div class="typing-indicator">🎵 Assistant is thinking <div class="dot"></div><div class="dot"></div><div class="dot"></div></div>', unsafe_allow_html=True)

                answer, tools_used, react_result = get_answer(user_input.strip(), use_react_agent=use_react_agent)
                conversation_manager.add_message("assistant", answer, tools_used, react_result)
                st.rerun()

        # Example chips
        st.markdown("### 💭 Quick Questions")
        st.markdown('<div class="example-chips">', unsafe_allow_html=True)

        examples = [
            "What is carnatic music?",
            "Tell me about raga Mayamalavagowla",
            "Explain melakarta",
            "Raga Bhairavi characteristics",
        ]

        for example in examples:
            if st.button(example, key=f"ex_{example}"):
                conversation_manager.add_message("user", example)
                answer, tools_used, react_result = get_answer(example, use_react_agent=use_react_agent)
                conversation_manager.add_message("assistant", answer, tools_used)
                st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)

    # Footer
    st.markdown("---")
    st.markdown(
        "<p style='text-align: center; color: #666;'>🎵 Powered by LangChain, Groq, and your Carnatic music knowledge base 🎵</p>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
