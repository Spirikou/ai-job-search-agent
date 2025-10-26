"""
AI Job Search Agent - Streamlit User Interface
Streamlined workflow: Upload Resume → Select Research Type → Results → Chat
"""

import streamlit as st
import json
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any
import sys

# Add the src directory to the path
src_path = Path(__file__).parent
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

# Import only the functions we actually use from agent_core
from agent_core import (
    convert_resume, load_resume_data, run_agent_with_memory
)

# ============================================================================
# STREAMLIT CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="AI Job Search Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS with Night Mode Support
st.markdown("""
<style>
    :root {
        --bg-primary: #0a0a0a;
        --bg-secondary: #1a1a1a;
        --bg-tertiary: #2a2a2a;
        --text-primary: #ffffff;
        --text-secondary: #e0e0e0;
        --text-muted: #c0c0c0;
        --accent-primary: #00d4ff;
        --accent-secondary: #ff6b35;
        --accent-success: #00ff88;
        --accent-warning: #ffaa00;
        --accent-info: #00aaff;
        --border-color: #333333;
        --shadow: 0 4px 12px rgba(0, 212, 255, 0.1);
    }
    
    .stApp {
        background-color: var(--bg-primary);
        color: var(--text-primary);
    }
    
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: var(--accent-primary);
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 0 0 20px rgba(0, 212, 255, 0.3);
    }
    
    .chat-message {
        padding: 1rem;
        border-radius: 0.8rem;
        margin: 0.5rem 0;
        border-left: 4px solid var(--accent-primary);
        background-color: var(--bg-secondary);
        box-shadow: var(--shadow);
    }
    
    .user-message {
        background-color: var(--bg-tertiary);
        border-left-color: var(--accent-primary);
    }
    
    .assistant-message {
        background-color: var(--bg-secondary);
        border-left-color: var(--accent-success);
    }
    
    .system-message {
        background-color: var(--bg-tertiary);
        border-left-color: var(--accent-info);
    }
    
    .job-card {
        background-color: var(--bg-secondary);
        padding: 1.5rem;
        border-radius: 0.8rem;
        border: 1px solid var(--border-color);
        margin: 0.5rem 0;
        box-shadow: var(--shadow);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    
    .job-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(0, 212, 255, 0.2);
    }
    
    .status-indicator {
        padding: 0.8rem 1.2rem;
        border-radius: 0.6rem;
        margin: 0.5rem 0;
        font-weight: bold;
        border: 1px solid var(--border-color);
    }
    
    .status-success {
        background-color: rgba(0, 255, 136, 0.1);
        color: var(--accent-success);
        border-color: var(--accent-success);
    }
    
    .status-warning {
        background-color: rgba(255, 170, 0, 0.1);
        color: var(--accent-warning);
        border-color: var(--accent-warning);
    }
    
    .status-info {
        background-color: rgba(0, 170, 255, 0.1);
        color: var(--accent-info);
        border-color: var(--accent-info);
    }
    
    .workflow-step {
        background-color: var(--bg-secondary);
        padding: 1.5rem;
        border-radius: 0.8rem;
        border: 2px solid var(--border-color);
        margin: 1rem 0;
        transition: all 0.3s ease;
    }
    
    .step-active {
        border-color: var(--accent-primary);
        background-color: rgba(0, 212, 255, 0.1);
        box-shadow: 0 0 20px rgba(0, 212, 255, 0.2);
    }
    
    .step-completed {
        border-color: var(--accent-success);
        background-color: rgba(0, 255, 136, 0.1);
    }
    
    .sidebar .workflow-step {
        margin: 0.5rem 0;
        padding: 1rem;
    }
    
    .stTextInput > div > div > input {
        background-color: var(--bg-secondary);
        color: var(--text-primary);
        border-color: var(--border-color);
    }
    
    .stTextArea > div > div > textarea {
        background-color: var(--bg-secondary);
        color: var(--text-primary);
        border-color: var(--border-color);
    }
    
    .stSelectbox > div > div > select {
        background-color: var(--bg-secondary);
        color: var(--text-primary);
        border-color: var(--border-color);
    }
    
    .stButton > button {
        background-color: var(--accent-primary);
        color: var(--bg-primary);
        border: none;
        border-radius: 0.5rem;
        font-weight: bold;
        transition: all 0.2s ease;
    }
    
    .stButton > button:hover {
        background-color: var(--accent-secondary);
        transform: translateY(-1px);
        box-shadow: var(--shadow);
    }
    
    .stFileUploader > div {
        background-color: var(--bg-secondary);
        border-color: var(--border-color);
    }
    
    .stExpander > div {
        background-color: var(--bg-secondary);
        border-color: var(--border-color);
    }
    
    .stExpander > div > div {
        color: var(--text-primary);
    }
    
    /* Improve text visibility for various elements */
    .stMarkdown p {
        color: var(--text-secondary);
    }
    
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4, .stMarkdown h5, .stMarkdown h6 {
        color: var(--text-primary);
    }
    
    .stMarkdown strong {
        color: var(--text-primary);
    }
    
    .stMarkdown em {
        color: var(--text-muted);
    }
    
    .stMarkdown ul, .stMarkdown ol {
        color: var(--text-secondary);
    }
    
    .stMarkdown li {
        color: var(--text-secondary);
    }
    
    .stMarkdown blockquote {
        color: var(--text-muted);
        border-left-color: var(--accent-primary);
    }
    
    .stMarkdown code {
        background-color: var(--bg-tertiary);
        color: var(--accent-primary);
        border: 1px solid var(--border-color);
    }
    
    .stMarkdown pre {
        background-color: var(--bg-tertiary);
        color: var(--text-primary);
        border: 1px solid var(--border-color);
    }
    
    .stMarkdown pre code {
        background-color: transparent;
        border: none;
    }
    
    /* Improve visibility for Streamlit widgets */
    .stSelectbox label, .stTextInput label, .stTextArea label, .stFileUploader label {
        color: var(--text-primary) !important;
    }
    
    .stSelectbox div[data-baseweb="select"] {
        color: var(--text-primary);
    }
    
    .stSelectbox div[data-baseweb="select"] span {
        color: var(--text-primary);
    }
    
    .stTextInput div[data-baseweb="input"] {
        color: var(--text-primary);
    }
    
    .stTextArea div[data-baseweb="textarea"] {
        color: var(--text-primary);
    }
    
    .stFileUploader div[data-testid="stFileUploader"] {
        color: var(--text-primary);
    }
    
    .stFileUploader div[data-testid="stFileUploader"] label {
        color: var(--text-primary) !important;
    }
    
    /* Improve visibility for help text */
    .stMarkdown .help {
        color: var(--text-muted) !important;
    }
    
    /* Improve visibility for info boxes */
    .stInfo {
        background-color: rgba(0, 170, 255, 0.1);
        border: 1px solid var(--accent-info);
        color: var(--text-primary);
    }
    
    .stSuccess {
        background-color: rgba(0, 255, 136, 0.1);
        border: 1px solid var(--accent-success);
        color: var(--text-primary);
    }
    
    .stWarning {
        background-color: rgba(255, 170, 0, 0.1);
        border: 1px solid var(--accent-warning);
        color: var(--text-primary);
    }
    
    .stError {
        background-color: rgba(255, 107, 53, 0.1);
        border: 1px solid var(--accent-secondary);
        color: var(--text-primary);
    }
    
    /* Improve visibility for sidebar */
    .css-1d391kg {
        background-color: var(--bg-secondary);
        color: var(--text-primary);
    }
    
    .css-1d391kg .stMarkdown {
        color: var(--text-primary);
    }
    
    .css-1d391kg .stMarkdown p {
        color: var(--text-secondary);
    }
    
    /* Improve visibility for main content area */
    .main .block-container {
        color: var(--text-primary);
    }
    
    .main .block-container .stMarkdown {
        color: var(--text-primary);
    }
    
    .main .block-container .stMarkdown p {
        color: var(--text-secondary);
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

def initialize_session_state():
    """Initialize session state variables."""
    if 'resume_loaded' not in st.session_state:
        st.session_state.resume_loaded = False
    if 'resume_data' not in st.session_state:
        st.session_state.resume_data = None
    if 'original_resume_name' not in st.session_state:
        st.session_state.original_resume_name = None
    if 'jobs_data' not in st.session_state:
        st.session_state.jobs_data = []
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'workflow_step' not in st.session_state:
        st.session_state.workflow_step = "resume_check"  # resume_check, preferences, prompt, search, results, chat
    if 'preferences' not in st.session_state:
        st.session_state.preferences = None
    if 'job_search_prompt' not in st.session_state:
        st.session_state.job_search_prompt = None
    if 'clear_input' not in st.session_state:
        st.session_state.clear_input = False

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_latest_cv_files():
    """Get the latest CV files (PDF and JSON) with error handling."""
    try:
        data_dir = Path("data")
        if not data_dir.exists():
            return None, None, "Data directory not found"
        
        # Get latest PDF file
        pdf_files = list(data_dir.glob("Resume_*.pdf"))
        latest_pdf = max(pdf_files, key=lambda x: x.stat().st_mtime) if pdf_files else None
        
        # Get latest JSON file
        json_files = list(data_dir.glob("CV_*.json"))
        latest_json = max(json_files, key=lambda x: x.stat().st_mtime) if json_files else None
        
        return latest_pdf, latest_json, None
        
    except Exception as e:
        return None, None, f"Error accessing CV files: {e}"

def create_download_button(file_path: Path, file_type: str, label: str, key_suffix: str = ""):
    """Create a download button with proper error handling."""
    if not file_path or not file_path.exists():
        st.info(f"📄 No {file_type} file available")
        return False
    
    try:
        with open(file_path, "rb") as file:
            file_data = file.read()
        
        mime_type = "application/pdf" if file_type.lower() == "pdf" else "application/json"
        
        st.download_button(
            label=label,
            data=file_data,
            file_name=file_path.name,
            mime=mime_type,
            key=f"download_{file_type.lower()}_{key_suffix}_{datetime.now().timestamp()}",
            use_container_width=True,
            help=f"Download the {file_type} file"
        )
        return True
        
    except Exception as e:
        st.error(f"❌ Error reading {file_type} file: {e}")
        return False

def upload_resume_file(uploaded_file) -> bool:
    """Upload and process resume PDF file with any name, rename to resume.pdf."""
    try:
        # Validate file type
        if uploaded_file.type != "application/pdf":
            st.error("Please upload a PDF file.")
            return False
        
        # Get original filename for display
        original_filename = uploaded_file.name
        
        # Save uploaded file as resume.pdf (standardized name for backend)
        resume_path = Path("data") / "resume.pdf"
        resume_path.parent.mkdir(exist_ok=True)
        
        # Remove existing resume.pdf if it exists
        if resume_path.exists():
            resume_path.unlink()
        
        with open(resume_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Verify file was saved
        if not resume_path.exists():
            st.error("❌ Failed to save resume file.")
            return False
        
        # Convert resume
        with st.spinner(f"Processing {original_filename}..."):
            convert_resume()
        
        # Load resume data
        resume_data, resume_text = load_resume_data()
        st.session_state.resume_data = resume_data
        st.session_state.resume_loaded = True
        st.session_state.workflow_step = "preferences"
        
        # Store original filename for reference
        st.session_state.original_resume_name = original_filename
        
        return True
    except Exception as e:
        st.error(f"Error processing resume: {e}")
        return False

def check_resume_file() -> bool:
    """Check if resume.pdf exists in data folder and load it."""
    try:
        resume_path = Path("data") / "resume.pdf"
        if not resume_path.exists():
            return False
        
        # Convert resume
        with st.spinner("Converting resume to structured data..."):
            convert_resume()
        
        # Load resume data
        resume_data, resume_text = load_resume_data()
        st.session_state.resume_data = resume_data
        st.session_state.resume_loaded = True
        st.session_state.workflow_step = "preferences"
        
        # Set a generic name for existing resume
        st.session_state.original_resume_name = "resume.pdf"
        
        return True
    except Exception as e:
        st.error(f"Error loading resume: {e}")
        return False

def load_preferences() -> Dict[str, Any]:
    """Load preferences from preferences.json file."""
    try:
        preferences_path = Path("data") / "preferences.json"
        if preferences_path.exists():
            with open(preferences_path, 'r') as f:
                return json.load(f)
        else:
            # Return default preferences
            return [{
                "roles": ["AI Engineer", "Machine Learning Engineer", "Data Scientist"],
                "location": ["London", "Remote"],
                "companies": ["Google", "Microsoft", "OpenAI"]
            }]
    except Exception as e:
        st.error(f"Error loading preferences: {e}")
        return [{
            "roles": ["AI Engineer", "Machine Learning Engineer", "Data Scientist"],
            "location": ["London", "Remote"],
            "companies": ["Google", "Microsoft", "OpenAI"]
        }]

def save_preferences(preferences: Dict[str, Any]) -> bool:
    """Save preferences to preferences.json file."""
    try:
        preferences_path = Path("data") / "preferences.json"
        preferences_path.parent.mkdir(exist_ok=True)
        
        with open(preferences_path, 'w') as f:
            json.dump(preferences, f, indent=2)
        
        return True
    except Exception as e:
        st.error(f"Error saving preferences: {e}")
        return False

def get_default_job_search_prompt() -> str:
    """Get the default job search prompt."""
    return """Find AI related job openings in London, UK that were published in the last 24 hours. Only include jobs listed on LinkedIn, Indeed and Google. 

Then analyze these jobs and find the best matches based on my resume and preferences using similarity matching with keyword reranking. 

Please display the results in a clean, readable format (NOT a table). For each job, show:
- Job Title at Company Name
- Location
- Date Posted: [publishing date in YYYY-MM-DD format]
- Match Score: [overall score as percentage]
- Job URL: [link]

Show me the top 20 most relevant job opportunities with their detailed analysis in this format."""

def display_chat_message(message: Dict[str, Any]):
    """Display a chat message with appropriate styling."""
    role = message.get('role', 'user')
    content = message.get('content', '')
    timestamp = message.get('timestamp', '')
    
    if role == 'user':
        st.markdown(f"""
        <div class="chat-message user-message">
            <strong>You:</strong> {content}
        </div>
        """, unsafe_allow_html=True)
    elif role == 'assistant':
        # Check if this contains CV content and display it specially
        if "**CV Content:**" in content:
            # Split content to separate message from CV content
            message_part, cv_content = content.split("**CV Content:**", 1)
            message_part = message_part.strip()
            cv_content = cv_content.strip()
            
            # Display the message part
            st.markdown(f"""
            <div class="chat-message assistant-message">
                <strong>AI Assistant:</strong> {message_part}
            </div>
            """, unsafe_allow_html=True)
            
            # Display CV content using Streamlit's native markdown rendering
            if cv_content:
                st.markdown("### 📄 Customized CV")
                
                # Use Streamlit's native markdown with custom styling
                st.markdown(f"""
                <div style='
                    background-color: var(--bg-secondary); 
                    padding: 2rem; 
                    border-radius: 0.8rem; 
                    border: 2px solid var(--accent-success); 
                    color: var(--text-primary); 
                    font-family: "Segoe UI", Arial, sans-serif;
                    line-height: 1.6;
                    box-shadow: 0 0 20px rgba(0, 255, 136, 0.2);
                    margin: 1rem 0;
                    max-height: 500px;
                    overflow-y: auto;
                '>
                """, unsafe_allow_html=True)
                
                # Render the CV content as markdown (this will properly format headers, bullets, etc.)
                st.markdown(cv_content)
                
                st.markdown("</div>", unsafe_allow_html=True)
                
                # Add download buttons for CV files
                col1, col2 = st.columns(2)
                
                with col1:
                    latest_pdf, latest_json, error = get_latest_cv_files()
                    if error:
                        st.error(f"❌ {error}")
                    else:
                        create_download_button(
                            latest_pdf,
                            "PDF",
                            "📥 Download PDF Resume",
                            "chat"
                        )
                
                with col2:
                    create_download_button(
                        latest_json,
                        "JSON",
                        "📥 Download JSON Data",
                        "chat"
                    )
        else:
            # Regular message - let Streamlit handle markdown naturally
            st.markdown(f"""
            <div class="chat-message assistant-message">
                <strong>AI Assistant:</strong> 
            </div>
            """, unsafe_allow_html=True)
            # Display content as markdown to handle any formatting
            st.markdown(content)
    elif role == 'system':
        st.markdown(f"""
        <div class="chat-message system-message">
            <strong>System:</strong> {content}
        </div>
        """, unsafe_allow_html=True)

def display_job_results(jobs_data: List[Dict[str, Any]]):
    """Display job results in a simplified format."""
    if not jobs_data:
        return
    
    st.markdown("### 📋 Job Search Results")
    
    # Create a simple list format
    job_list = []
    
    for i, job in enumerate(jobs_data):
        # Format score as percentage
        score = job.get('score', 0)
        if isinstance(score, (int, float)):
            # Handle both decimal (0.85) and percentage (85) formats
            if score <= 1:
                score_display = f"{score * 100:.1f}%"
            else:
                score_display = f"{score:.1f}%"
        else:
            score_display = str(score)
        
        # Format date
        date_posted = job.get('date_posted', 'N/A')
        if date_posted and date_posted != 'N/A':
            try:
                from datetime import datetime
                if isinstance(date_posted, str):
                    # Try to parse and format the date
                    date_obj = datetime.strptime(date_posted, '%Y-%m-%d')
                    date_display = date_obj.strftime('%B %d, %Y')
                else:
                    date_display = str(date_posted)
            except:
                date_display = str(date_posted)
        else:
            date_display = 'N/A'
        
        # Create job entry with clickable link
        job_url = job.get('url', 'N/A')
        if job_url and job_url != 'N/A':
            job_entry = f"{i+1}. **[{job.get('title', 'N/A')}]({job_url})** at {job.get('company', 'N/A')} | {job.get('location', 'N/A')} | {date_display} | Match: {score_display}"
        else:
            job_entry = f"{i+1}. **{job.get('title', 'N/A')}** at {job.get('company', 'N/A')} | {job.get('location', 'N/A')} | {date_display} | Match: {score_display}"
        job_list.append(job_entry)
    
    # Display all jobs in a single cell
    st.markdown("**All Job Opportunities:**")
    st.markdown("\n".join(job_list))






# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    """Main application function."""
    # Initialize session state
    initialize_session_state()
    
    # Header
    st.markdown('<h1 class="main-header">🤖 AI Job Search Agent</h1>', unsafe_allow_html=True)
    
    # Main content area - full width when not in chat mode
    if st.session_state.workflow_step == "chat":
        col1, col2 = st.columns([2, 1])
    else:
        col1, col2 = st.columns([3, 1])
    
    with col1:
        # Chat interface (only available after workflow completion)
        if st.session_state.workflow_step == "chat":
            st.markdown("### 💬 AI Assistant Chat")
            
            # Add helpful tips for CV creation and comparison
            if not st.session_state.chat_history:
                st.info("""
                💡 **Tips for CV Management:**
                - **Create CVs:** "Create a CV for Software Engineer at Google"
                - **Job-specific CVs:** "Create a CV for job #3" or "Customize my CV for the Microsoft job"
                - **Compare CVs:** "Compare my original CV with the customized version"
                - **Analyze Changes:** "Explain the differences between my CVs"
                - **List CVs:** "Show me all my CV files"
                - **Get Details:** "What changes were made to my CV for the Google job?"
                """)
                
                with st.expander("🐛 Debugging & Logs"):
                    st.markdown("""
                    **To view detailed logs and debug errors:**
                    1. Open the terminal/console where Streamlit is running
                    2. Look for print statements showing each step (🎯, 📥, 🧠, etc.)
                    3. Errors will show which step failed and full tracebacks
                    4. Timestamps show how long each step takes
                    
                    **Common Error Messages:**
                    - ❌ ERROR in [step_name]: Shows exactly which step failed
                    - Time elapsed: Shows processing time for performance analysis
                    - Full Traceback: Complete error details for debugging
                    """)
            
            # Display chat history
            for message in st.session_state.chat_history:
                display_chat_message(message)
            
            # Chat input with Enter key support
            # Check if we need to clear the input
            input_value = "" if st.session_state.get('clear_input', False) else None
            if st.session_state.get('clear_input', False):
                st.session_state.clear_input = False
            
            user_input = st.text_input(
                "Ask me anything about jobs, CV customization, or career advice:", 
                value=input_value,
                key="chat_input",
                on_change=lambda: setattr(st.session_state, 'send_message', True)
            )
            
            # Check if Enter was pressed or Send button clicked
            send_message = st.button("Send", key="send_chat") or st.session_state.get('send_message', False)
            
            if send_message and user_input:
                # Store the user input before clearing
                current_input = user_input
                
                # Reset the send_message flag
                st.session_state.send_message = False
                
                # Add user message to history
                st.session_state.chat_history.append({
                    'role': 'user',
                    'content': current_input,
                    'timestamp': datetime.now().isoformat()
                })
                
                # Get AI response for all requests using the agent
                with st.spinner("AI is thinking..."):
                    try:
                        response = run_agent_with_memory(current_input)
                        
                        # Add assistant response to history
                        st.session_state.chat_history.append({
                            'role': 'assistant',
                            'content': response,
                            'timestamp': datetime.now().isoformat()
                        })
                        
                        # Clear the input by rerunning with a flag
                        st.session_state.clear_input = True
                        st.rerun()
                        
                    except Exception as e:
                        import traceback
                        error_details = f"""
**Error Details:**
- Error Type: {type(e).__name__}
- Error Message: {str(e)}
- Full Traceback:
```
{traceback.format_exc()}
```

**Troubleshooting:**
1. Check the console/terminal where Streamlit is running for detailed logs
2. Look for print statements showing which step failed
3. Common issues: Network timeout, API rate limits, or LLM processing timeout
"""
                        st.session_state.chat_history.append({
                            'role': 'assistant',
                            'content': f"Error getting AI response: {e}\n{error_details}",
                            'timestamp': datetime.now().isoformat()
                        })
                        st.session_state.clear_input = True
                        st.rerun()
        else:
            # Show workflow progress and inputs/results
            st.markdown("### 📊 Workflow Progress")
            
            # Show current inputs
            if st.session_state.resume_loaded:
                st.markdown("#### ✅ Resume Loaded")
                if st.session_state.original_resume_name:
                    st.success(f"Resume '{st.session_state.original_resume_name}' has been successfully loaded and processed.")
                else:
                    st.success("Resume has been successfully loaded and processed.")
            
            if st.session_state.preferences:
                st.markdown("#### ✅ Preferences Set")
                prefs = st.session_state.preferences[0]
                
                # Display preferences in a clean, expanded format
                st.markdown("**Current Preferences:**")
                
                col_pref1, col_pref2, col_pref3 = st.columns(3)
                
                with col_pref1:
                    st.markdown(f"""
                    <div style='
                        background-color: var(--bg-tertiary); 
                        padding: 1.2rem; 
                        border-radius: 0.8rem; 
                        border: 1px solid var(--border-color); 
                        color: var(--text-secondary); 
                        font-family: "Segoe UI", Arial, sans-serif;
                        line-height: 1.6;
                        box-shadow: var(--shadow);
                        margin: 0.5rem 0;
                        height: 200px;
                        overflow-y: auto;
                    '>
                    <h4 style='color: var(--accent-primary); margin-top: 0;'>🎯 Job Roles</h4>
                    {''.join([f"• {role}<br>" for role in prefs.get('roles', [])])}
                    </div>
                    """, unsafe_allow_html=True)
                
                with col_pref2:
                    st.markdown(f"""
                    <div style='
                        background-color: var(--bg-tertiary); 
                        padding: 1.2rem; 
                        border-radius: 0.8rem; 
                        border: 1px solid var(--border-color); 
                        color: var(--text-secondary); 
                        font-family: "Segoe UI", Arial, sans-serif;
                        line-height: 1.6;
                        box-shadow: var(--shadow);
                        margin: 0.5rem 0;
                        height: 200px;
                        overflow-y: auto;
                    '>
                    <h4 style='color: var(--accent-primary); margin-top: 0;'>📍 Locations</h4>
                    {''.join([f"• {location}<br>" for location in prefs.get('location', [])])}
                    </div>
                    """, unsafe_allow_html=True)
                
                with col_pref3:
                    st.markdown(f"""
                    <div style='
                        background-color: var(--bg-tertiary); 
                        padding: 1.2rem; 
                        border-radius: 0.8rem; 
                        border: 1px solid var(--border-color); 
                        color: var(--text-secondary); 
                        font-family: "Segoe UI", Arial, sans-serif;
                        line-height: 1.6;
                        box-shadow: var(--shadow);
                        margin: 0.5rem 0;
                        height: 200px;
                        overflow-y: auto;
                    '>
                    <h4 style='color: var(--accent-primary); margin-top: 0;'>🏢 Companies</h4>
                    {''.join([f"• {company}<br>" for company in prefs.get('companies', [])])}
                    </div>
                    """, unsafe_allow_html=True)
            
            if st.session_state.job_search_prompt:
                st.markdown("#### ✅ Search Prompt Ready")
                
                # Clean and format the prompt for better readability
                cleaned_prompt = st.session_state.job_search_prompt.strip()
                # Remove excessive line breaks and normalize spacing
                cleaned_prompt = '\n'.join(line.strip() for line in cleaned_prompt.split('\n') if line.strip())
                
                # Display search prompt in a clean, readable format
                st.markdown("**Job Search Prompt:**")
                st.markdown(f"""
                <div style='
                    background-color: var(--bg-tertiary); 
                    padding: 1.5rem; 
                    border-radius: 0.8rem; 
                    border: 2px solid var(--accent-primary); 
                    color: var(--text-secondary); 
                    font-family: "Segoe UI", Arial, sans-serif;
                    line-height: 1.6;
                    white-space: pre-line;
                    box-shadow: 0 0 20px rgba(0, 212, 255, 0.2);
                    margin: 0.5rem 0;
                    max-height: 250px;
                    overflow-y: auto;
                '>{cleaned_prompt}</div>
                """, unsafe_allow_html=True)
            
            if st.session_state.jobs_data:
                st.markdown("#### ✅ Job Search Results")
                st.success(f"Found {len(st.session_state.jobs_data)} job matches!")
                display_job_results(st.session_state.jobs_data)
            
            # Show current step content that user needs to complete
            if st.session_state.workflow_step == "resume_check":
                st.markdown("---")
                st.markdown("### 📄 Step 1: Resume Upload")
                st.markdown("Upload your resume in PDF format. You can use any filename - it will be automatically processed.")
                
                # File upload with custom styling
                st.markdown("""
                <style>
                .stFileUploader > div {
                    background-color: var(--bg-secondary);
                    border: 2px dashed var(--border-color);
                    border-radius: 0.8rem;
                    padding: 1rem;
                    text-align: center;
                    transition: all 0.3s ease;
                    min-height: 80px;
                }
                .stFileUploader > div:hover {
                    border-color: var(--accent-primary);
                    background-color: rgba(0, 212, 255, 0.05);
                }
                .stFileUploader label {
                    font-size: 0.9rem !important;
                    color: #000000 !important;
                }
                </style>
                """, unsafe_allow_html=True)
                
                uploaded_file = st.file_uploader(
                    "📄 Upload Resume (PDF)",
                    type=['pdf'],
                    help="Drag & drop your PDF resume here or click to browse",
                    label_visibility="collapsed"
                )
                
                if uploaded_file is not None:
                    if st.button("📤 Upload and Process Resume"):
                        if upload_resume_file(uploaded_file):
                            original_name = uploaded_file.name
                            st.success(f"✅ Resume '{original_name}' loaded successfully!")
                            st.session_state.chat_history.append({
                                'role': 'system',
                                'content': f"✅ Resume '{original_name}' loaded successfully! Please review your preferences.",
                                'timestamp': datetime.now().isoformat()
                            })
                            st.rerun()
                
                # Alternative: Check for existing resume
                st.markdown("---")
                st.markdown("**Or check for existing resume:**")
                if st.button("🔍 Check for Existing Resume"):
                    if check_resume_file():
                        st.success("✅ Resume found and loaded successfully!")
                        st.session_state.chat_history.append({
                            'role': 'system',
                            'content': "✅ Resume found and loaded successfully! Please review your preferences.",
                            'timestamp': datetime.now().isoformat()
                        })
                        st.rerun()
                    else:
                        st.error("❌ Resume not found! Please upload a resume above.")
            
            elif st.session_state.workflow_step == "preferences":
                st.markdown("---")
                st.markdown("### ⚙️ Step 2: Preferences")
                
                # Load current preferences
                if st.session_state.preferences is None:
                    st.session_state.preferences = load_preferences()
                
                current_prefs = st.session_state.preferences[0] if st.session_state.preferences else {}
                
                st.markdown("#### Current Preferences:")
                
                # Display current preferences
                col_pref1, col_pref2 = st.columns(2)
                
                with col_pref1:
                    st.markdown("**Job Roles:**")
                    for role in current_prefs.get('roles', []):
                        st.markdown(f"- {role}")
                    
                    st.markdown("**Locations:**")
                    for location in current_prefs.get('location', []):
                        st.markdown(f"- {location}")
                
                with col_pref2:
                    st.markdown("**Companies:**")
                    for company in current_prefs.get('companies', []):
                        st.markdown(f"- {company}")
                
                st.markdown("#### Modify Preferences:")
                
                # Edit preferences
                new_roles = st.text_area(
                    "Job Roles (one per line)",
                    value="\n".join(current_prefs.get('roles', [])),
                    help="Enter job roles, one per line"
                )
                
                new_locations = st.text_area(
                    "Locations (one per line)",
                    value="\n".join(current_prefs.get('location', [])),
                    help="Enter preferred locations, one per line"
                )
                
                new_companies = st.text_area(
                    "Companies (one per line)",
                    value="\n".join(current_prefs.get('companies', [])),
                    help="Enter preferred companies, one per line"
                )
                
                if st.button("💾 Save Preferences"):
                    # Update preferences
                    updated_prefs = [{
                        "roles": [role.strip() for role in new_roles.split('\n') if role.strip()],
                        "location": [loc.strip() for loc in new_locations.split('\n') if loc.strip()],
                        "companies": [comp.strip() for comp in new_companies.split('\n') if comp.strip()]
                    }]
                    
                    if save_preferences(updated_prefs):
                        st.session_state.preferences = updated_prefs
                        st.session_state.workflow_step = "prompt"
                        st.success("✅ Preferences saved successfully!")
                        st.session_state.chat_history.append({
                            'role': 'system',
                            'content': "✅ Preferences saved! Please review the job search prompt.",
                            'timestamp': datetime.now().isoformat()
                        })
                        st.rerun()
            
            elif st.session_state.workflow_step == "prompt":
                st.markdown("---")
                st.markdown("### 📝 Step 3: Job Search Prompt")
                
                # Load or create default prompt
                if st.session_state.job_search_prompt is None:
                    st.session_state.job_search_prompt = get_default_job_search_prompt()
                
                st.markdown("Review and modify the job search prompt:")
                
                # Add visual indication for editing area
                st.markdown("""
                <div style='
                    background: linear-gradient(90deg, var(--accent-primary), transparent);
                    height: 3px;
                    border-radius: 2px;
                    margin: 1rem 0;
                    box-shadow: 0 0 10px rgba(0, 212, 255, 0.5);
                '></div>
                """, unsafe_allow_html=True)
                
                st.markdown("**✏️ Edit Area - Make your changes below:**")
                
                new_prompt = st.text_area(
                    "Job Search Prompt",
                    value=st.session_state.job_search_prompt,
                    height=300,
                    help="Modify the job search prompt as needed",
                    key="prompt_editor"
                )
                
                # Add visual indication after editing area
                st.markdown("""
                <div style='
                    background: linear-gradient(90deg, transparent, var(--accent-primary));
                    height: 3px;
                    border-radius: 2px;
                    margin: 1rem 0;
                    box-shadow: 0 0 10px rgba(0, 212, 255, 0.5);
                '></div>
                """, unsafe_allow_html=True)
                
                if st.button("💾 Save Prompt"):
                    st.session_state.job_search_prompt = new_prompt
                    st.session_state.workflow_step = "search"
                    st.success("✅ Prompt saved successfully!")
                    st.session_state.chat_history.append({
                        'role': 'system',
                        'content': "✅ Job search prompt saved! Ready to start job search.",
                        'timestamp': datetime.now().isoformat()
                    })
                    st.rerun()
            
            elif st.session_state.workflow_step == "search":
                st.markdown("---")
                st.markdown("### 🔍 Step 4: Job Search")
                st.markdown("Ready to search for jobs using your preferences and prompt.")
                
                if st.button("🚀 Start Job Search"):
                    with st.spinner("Searching for jobs..."):
                        try:
                            # Use the agent to perform job search
                            response = run_agent_with_memory(st.session_state.job_search_prompt)
                            
                            # Add to chat history
                            st.session_state.chat_history.append({
                                'role': 'assistant',
                                'content': response,
                                'timestamp': datetime.now().isoformat()
                            })
                            
                            # Try to load job results from the generated file
                            try:
                                job_results_path = Path("data") / "job_results_v5.json"
                                if job_results_path.exists():
                                    with open(job_results_path, 'r', encoding='utf-8') as f:
                                        job_data = json.load(f)
                                        st.session_state.jobs_data = job_data.get('jobs', [])
                            except Exception as e:
                                print(f"⚠️ Could not load job results: {e}")
                                st.session_state.jobs_data = []
                            
                            # Automatically move to results step and display jobs
                            st.session_state.workflow_step = "results"
                            st.rerun()
                            
                        except Exception as e:
                            st.session_state.chat_history.append({
                                'role': 'assistant',
                                'content': f"❌ Error during job search: {e}",
                                'timestamp': datetime.now().isoformat()
                            })
                            st.rerun()
            
            elif st.session_state.workflow_step == "results":
                st.markdown("---")
                st.markdown("### 📋 Step 5: Job Results")
                
                # Always show results if available
                if st.session_state.jobs_data:
                    st.success(f"✅ Found {len(st.session_state.jobs_data)} job matches!")
                    display_job_results(st.session_state.jobs_data)
                    
                    st.markdown("---")
                    st.markdown("### 💬 Ready to Chat!")
                    st.markdown("You can now:")
                    st.markdown("- Ask questions about specific jobs")
                    st.markdown("- Request customized CVs for job applications")
                    st.markdown("- Get career advice and insights")
                    st.markdown("- Analyze job requirements in detail")
                    
                    # Auto-transition to chat mode
                    st.session_state.workflow_step = "chat"
                    st.rerun()
                else:
                    st.info("Job search completed! Check the chat for results and use the chat to ask questions or create customized CVs.")
                    
                    # Auto-transition to chat mode
                    st.session_state.workflow_step = "chat"
                    st.rerun()
    
    # Right sidebar with workflow steps
    with col2:
        st.markdown("### 📋 Workflow Steps")
        
        # Step 1: Resume
        step_class = "step-completed" if st.session_state.resume_loaded else ("step-active" if st.session_state.workflow_step == "resume_check" else "")
        st.markdown(f'<div class="workflow-step {step_class}"><strong>1. Resume Upload</strong><br>{"✅ Complete" if st.session_state.resume_loaded else "⏳ Pending"}</div>', unsafe_allow_html=True)
        
        # Step 2: Preferences
        step_class = "step-completed" if st.session_state.preferences else ("step-active" if st.session_state.workflow_step == "preferences" else "")
        st.markdown(f'<div class="workflow-step {step_class}"><strong>2. Preferences</strong><br>{"✅ Complete" if st.session_state.preferences else "⏳ Pending"}</div>', unsafe_allow_html=True)
        
        # Step 3: Prompt
        step_class = "step-completed" if st.session_state.job_search_prompt else ("step-active" if st.session_state.workflow_step == "prompt" else "")
        st.markdown(f'<div class="workflow-step {step_class}"><strong>3. Search Prompt</strong><br>{"✅ Complete" if st.session_state.job_search_prompt else "⏳ Pending"}</div>', unsafe_allow_html=True)
        
        # Step 4: Search
        step_class = "step-completed" if st.session_state.jobs_data else ("step-active" if st.session_state.workflow_step == "search" else "")
        st.markdown(f'<div class="workflow-step {step_class}"><strong>4. Job Search</strong><br>{"✅ Complete" if st.session_state.jobs_data else "⏳ Pending"}</div>', unsafe_allow_html=True)
        
        # Step 5: Results
        step_class = "step-completed" if st.session_state.jobs_data else ("step-active" if st.session_state.workflow_step == "results" else "")
        st.markdown(f'<div class="workflow-step {step_class}"><strong>5. View Results</strong><br>{"✅ Complete" if st.session_state.jobs_data else "⏳ Pending"}</div>', unsafe_allow_html=True)
        
        # Step 6: Chat
        step_class = "step-active" if st.session_state.workflow_step == "chat" else ""
        st.markdown(f'<div class="workflow-step {step_class}"><strong>6. Chat & Customize</strong><br>💬 Available</div>', unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("### 📊 Status")
        
        if st.session_state.resume_loaded:
            st.markdown('<div class="status-indicator status-success">✅ Resume Loaded</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="status-indicator status-warning">⚠️ Resume Not Loaded</div>', unsafe_allow_html=True)
        
        if st.session_state.preferences:
            pref_count = len(st.session_state.preferences[0].get('roles', [])) if st.session_state.preferences else 0
            st.markdown(f'<div class="status-indicator status-success">✅ Preferences: {pref_count} roles</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="status-indicator status-info">ℹ️ No Preferences Set</div>', unsafe_allow_html=True)
        
        if st.session_state.job_search_prompt:
            st.markdown('<div class="status-indicator status-success">✅ Search Prompt Ready</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="status-indicator status-info">ℹ️ No Search Prompt</div>', unsafe_allow_html=True)
        
        if st.session_state.jobs_data:
            st.markdown(f'<div class="status-indicator status-success">✅ {len(st.session_state.jobs_data)} Jobs Found</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="status-indicator status-info">ℹ️ No Jobs Found</div>', unsafe_allow_html=True)
        
        st.markdown("### 💡 Tips")
        st.markdown("""
        - Upload your resume (PDF format)
        - Review and modify preferences
        - Customize the search prompt
        - Use chat to ask questions and create CVs
        """)

# ============================================================================
# RUN APPLICATION
# ============================================================================

if __name__ == "__main__":
    main()
