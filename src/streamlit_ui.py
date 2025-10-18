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
    convert_resume, load_resume_data, run_agent_with_memory,
    analyze_job_requirements, customize_cv, generate_cv_json, generate_cv_pdf, fetch_job_description
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
        # Check if this is a CV-related message and display it specially
        if any(keyword in content.lower() for keyword in ["customized cv", "cv content", "resume created", "**cv content:**"]):
            # Split content to separate the message from CV content
            lines = content.split('\n')
            message_part = []
            cv_content_part = []
            in_cv_section = False
            
            for line in lines:
                if "**CV Content:**" in line:
                    in_cv_section = True
                    continue
                elif in_cv_section:
                    cv_content_part.append(line)
                else:
                    message_part.append(line)
            
            message_text = '\n'.join(message_part).strip()
            cv_text = '\n'.join(cv_content_part).strip()
            
            st.markdown(f"""
            <div class="chat-message assistant-message">
                <strong>AI Assistant:</strong> {message_text}
            </div>
            """, unsafe_allow_html=True)
            
            if cv_text:
                # Enhanced CV display with better formatting
                st.markdown("### 📄 Customized CV")
                st.markdown(f"""
                <div style='
                    background-color: var(--bg-secondary); 
                    padding: 2rem; 
                    border-radius: 0.8rem; 
                    border: 2px solid var(--accent-success); 
                    color: var(--text-primary); 
                    font-family: "Segoe UI", Arial, sans-serif;
                    line-height: 1.6;
                    white-space: pre-wrap;
                    box-shadow: 0 0 20px rgba(0, 255, 136, 0.2);
                    margin: 1rem 0;
                    max-height: 500px;
                    overflow-y: auto;
                '>{cv_text}</div>
                """, unsafe_allow_html=True)
                
                # Add download buttons for CV files
                col1, col2 = st.columns(2)
                
                with col1:
                    # Look for PDF files
                    pdf_files = list(Path("data").glob("Resume_*.pdf"))
                    if pdf_files:
                        latest_pdf = max(pdf_files, key=lambda x: x.stat().st_mtime)
                        with open(latest_pdf, "rb") as pdf_file:
                            pdf_data = pdf_file.read()
                        
                        st.download_button(
                            label="📥 Download PDF",
                            data=pdf_data,
                            file_name=latest_pdf.name,
                            mime="application/pdf",
                            key=f"chat_download_pdf_{datetime.now().timestamp()}",
                            use_container_width=True
                        )
                
                with col2:
                    # Look for JSON files
                    json_files = list(Path("data").glob("CV_*.json"))
                    if json_files:
                        latest_json = max(json_files, key=lambda x: x.stat().st_mtime)
                        with open(latest_json, "rb") as json_file:
                            json_data = json_file.read()
                        
                        st.download_button(
                            label="📥 Download JSON",
                            data=json_data,
                            file_name=latest_json.name,
                            mime="application/json",
                            key=f"chat_download_json_{datetime.now().timestamp()}",
                            use_container_width=True
                        )
        else:
            st.markdown(f"""
            <div class="chat-message assistant-message">
                <strong>AI Assistant:</strong> {content}
            </div>
            """, unsafe_allow_html=True)
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

def create_cv_for_job(job: Dict[str, Any]):
    """Create customized CV for a specific job with enhanced progress indication and display."""
    job_title = job.get('title', 'N/A')
    company = job.get('company', 'N/A')
    job_url = job.get('url', None)
    
    # Create a more detailed progress indicator
    progress_container = st.container()
    with progress_container:
        st.markdown("### 📄 Generating Customized CV")
        
        # Progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Step 1: Fetch job description if URL provided
            job_description = None
            if job_url:
                status_text.text("📥 Fetching job description...")
                progress_bar.progress(10)
                job_description = fetch_job_description(job_url)
            
            # Step 2: Analyze job requirements
            status_text.text("🔍 Analyzing job requirements...")
            progress_bar.progress(25)
            job_requirements = analyze_job_requirements(
                job_description or f"Job title: {job_title} at {company}",
                job_title,
                company
            )
            
            # Step 3: Create customized CV
            status_text.text("✏️ Customizing resume content...")
            progress_bar.progress(50)
            customized_cv = customize_cv(job_requirements, job_title, company)
            
            if not customized_cv:
                raise Exception("CV customization failed")
            
            # Step 4: Generate JSON file
            status_text.text("📄 Generating JSON file...")
            progress_bar.progress(70)
            json_path = generate_cv_json(customized_cv, job_title, company)
            
            # Step 5: Generate PDF file
            status_text.text("📄 Generating PDF document...")
            progress_bar.progress(90)
            pdf_path = generate_cv_pdf(customized_cv, job_title, company, job_description)
            
            # Complete progress
            progress_bar.progress(100)
            status_text.text("🎉 CV generation completed!")
            time.sleep(1)
            
            # Clear progress indicators
            progress_bar.empty()
            status_text.empty()
            
            # Get the formatted CV content for display
            cv_content = get_formatted_cv_content(job_title, company, [Path(json_path)] if json_path else [])
            
            # Display the customized CV content
            st.markdown("### 📄 Customized CV Preview")
            st.markdown(f"""
            <div style='
                background-color: var(--bg-secondary); 
                padding: 2rem; 
                border-radius: 0.8rem; 
                border: 2px solid var(--accent-success); 
                color: var(--text-primary); 
                font-family: "Segoe UI", Arial, sans-serif;
                line-height: 1.6;
                white-space: pre-wrap;
                box-shadow: 0 0 20px rgba(0, 255, 136, 0.2);
                margin: 1rem 0;
                max-height: 500px;
                overflow-y: auto;
            '>{cv_content}</div>
            """, unsafe_allow_html=True)
            
            # Provide download options
            col1, col2 = st.columns(2)
            
            with col1:
                if pdf_path and Path(pdf_path).exists():
                    # Read PDF file for download
                    with open(pdf_path, "rb") as pdf_file:
                        pdf_data = pdf_file.read()
                    
                    st.download_button(
                        label="📥 Download PDF",
                        data=pdf_data,
                        file_name=Path(pdf_path).name,
                        mime="application/pdf",
                        key=f"download_pdf_{job_title}_{company}",
                        use_container_width=True
                    )
                else:
                    st.info("PDF not available")
            
            with col2:
                if json_path and Path(json_path).exists():
                    # Read JSON file for download
                    with open(json_path, "rb") as json_file:
                        json_data = json_file.read()
                    
                    st.download_button(
                        label="📥 Download JSON",
                        data=json_data,
                        file_name=Path(json_path).name,
                        mime="application/json",
                        key=f"download_json_{job_title}_{company}",
                        use_container_width=True
                    )
                else:
                    st.info("JSON not available")
            
            # Add to chat history with CV content
            chat_cv_content = f"✅ Customized CV created successfully for {job_title} at {company}!\n\n**CV Content:**\n{cv_content}\n\nDownload options are available above."
            
            st.session_state.chat_history.append({
                'role': 'assistant',
                'content': chat_cv_content,
                'timestamp': datetime.now().isoformat()
            })
            
            st.rerun()
            
        except Exception as e:
            # Clear progress indicators on error
            progress_bar.empty()
            status_text.empty()
            
            st.session_state.chat_history.append({
                'role': 'assistant',
                'content': f"❌ Error creating CV: {e}",
                'timestamp': datetime.now().isoformat()
            })
            st.rerun()

def get_formatted_cv_content(job_title: str, company: str, json_files: List[Path]) -> str:
    """Get the formatted CV content using LLM formatting."""
    if not json_files:
        return "CV content not available"
    
    try:
        # Get the latest JSON file
        latest_json = max(json_files, key=lambda x: x.stat().st_mtime)
        
        with open(latest_json, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        
        if 'cv_data' not in json_data:
            return "CV data not found in JSON file"
        
        cv_data = json_data['cv_data']
        
        # Use LLM to format the CV content for display
        from langchain_openai import ChatOpenAI
        
        llm_model = ChatOpenAI(model="gpt-4o-mini")
        
        prompt = f"""
You are a professional resume formatter. Convert this JSON CV data into a clean, readable format for chat display.

TARGET JOB: {job_title} at {company}

CV DATA (JSON):
{json.dumps(cv_data, indent=2)}

CRITICAL INSTRUCTIONS:
1. Convert the JSON data into a clean, readable CV format
2. Use markdown formatting for headers (## for main sections, ### for subsections)
3. Format experience with company, title, duration, and bullet points
4. Format skills in organized categories
5. Format education with degree, institution, and year
6. Do NOT include the raw JSON data
7. Do NOT include any analysis or commentary
8. Return ONLY the formatted CV text

EXAMPLE FORMAT:
## Summary
[Summary text here]

## Experience
### Company Name - Job Title
Duration: [Start Date - End Date]
• Achievement 1
• Achievement 2

## Skills
### Technical Skills
• Skill 1, Skill 2, Skill 3

## Education
### Degree Name
Institution Name, Year

Now format the provided CV data following this structure.
"""
        
        response = llm_model.invoke([
            {"role": "system", "content": "You are a professional resume formatter. Convert JSON CV data into clean, readable markdown format. Never include raw JSON in your response."},
            {"role": "user", "content": prompt}
        ])
        
        formatted_content = response.content.strip()
        
        # Validate that the response is not just JSON
        if formatted_content.startswith('{') or 'json' in formatted_content.lower():
            print("⚠️ LLM returned JSON instead of formatted text, using fallback")
            return build_readable_cv(cv_data)
        
        return formatted_content
        
    except Exception as e:
        print(f"❌ Error formatting CV content: {e}")
        # Fallback to basic formatting if LLM fails
        return build_readable_cv(cv_data)

def build_readable_cv(cv_data: Dict[str, Any]) -> str:
    """Build a readable version of the CV from JSON data."""
    cv_text = []
    
    # Add summary
    if cv_data.get('summary'):
        cv_text.append(f"## Summary\n{cv_data['summary']}\n")
    
    # Add experience
    if cv_data.get('experience'):
        cv_text.append("## Experience")
        for exp in cv_data['experience']:
            cv_text.append(f"\n### {exp.get('title', 'N/A')} at {exp.get('company', 'N/A')}")
            cv_text.append(f"**Duration:** {exp.get('duration', 'N/A')}")
            cv_text.append(f"**Location:** {exp.get('location', 'N/A')}")
            if exp.get('description'):
                cv_text.append("**Key Achievements:**")
                for desc in exp['description']:
                    cv_text.append(f"• {desc}")
            cv_text.append("")
    
    # Add education
    if cv_data.get('education'):
        cv_text.append("## Education")
        for edu in cv_data['education']:
            cv_text.append(f"\n### {edu.get('degree', 'N/A')}")
            cv_text.append(f"**Institution:** {edu.get('institution', 'N/A')}")
            cv_text.append(f"**Year:** {edu.get('year', 'N/A')}")
            if edu.get('location'):
                cv_text.append(f"**Location:** {edu.get('location', 'N/A')}")
        cv_text.append("")
    
    # Add skills
    if cv_data.get('skills'):
        cv_text.append("## Skills")
        for category, skills in cv_data['skills'].items():
            cv_text.append(f"\n### {category.title()}")
            cv_text.append(f"{', '.join(skills)}")
        cv_text.append("")
    
    # Add certifications
    if cv_data.get('certifications'):
        cv_text.append("## Certifications")
        for cert in cv_data['certifications']:
            cv_text.append(f"• {cert}")
        cv_text.append("")
    
    # Add projects
    if cv_data.get('projects'):
        cv_text.append("## Projects")
        for project in cv_data['projects']:
            cv_text.append(f"\n### {project.get('name', 'N/A')}")
            cv_text.append(f"{project.get('description', 'N/A')}")
            if project.get('technologies'):
                cv_text.append(f"**Technologies:** {', '.join(project['technologies'])}")
        cv_text.append("")
    
    return '\n'.join(cv_text)


def handle_cv_creation_request(user_input: str) -> bool:
    """Handle CV creation requests from chat input. Returns True if handled."""
    user_input_lower = user_input.lower()
    
    # Check for CV creation patterns
    cv_patterns = [
        "create cv", "create resume", "customize cv", "customize resume",
        "generate cv", "generate resume", "make cv", "make resume"
    ]
    
    if not any(pattern in user_input_lower for pattern in cv_patterns):
        return False
    
    # Try to extract job information from the user input
    # Look for job titles and companies in the input
    import re
    
    # Simple extraction patterns
    job_title_match = re.search(r'(?:for|as|to)\s+([^,\n]+?)(?:\s+at|\s+for|\s*$)', user_input, re.IGNORECASE)
    company_match = re.search(r'(?:at|for)\s+([^,\n]+?)(?:\s*$)', user_input, re.IGNORECASE)
    
    job_title = job_title_match.group(1).strip() if job_title_match else "Software Engineer"
    company = company_match.group(1).strip() if company_match else "Tech Company"
    
    # Create a mock job object for CV creation
    mock_job = {
        'title': job_title,
        'company': company,
        'url': None,
        'location': 'Remote',
        'score': 0.85
    }
    
    # Add a message to chat history indicating CV creation
    st.session_state.chat_history.append({
        'role': 'assistant',
        'content': f"🎯 I'll create a customized CV for **{job_title}** at **{company}**. Let me analyze the requirements and generate your personalized resume...",
        'timestamp': datetime.now().isoformat()
    })
    
    # Create the CV
    create_cv_for_job(mock_job)
    
    return True


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
                - **Compare CVs:** "Compare my original CV with the customized version"
                - **Analyze Changes:** "Explain the differences between my CVs"
                - **List CVs:** "Show me all my CV files"
                - **Get Details:** "What changes were made to my CV for the Google job?"
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
                
                # Check if this is a CV creation request first
                if handle_cv_creation_request(current_input):
                    # CV creation was handled, clear input and rerun
                    st.session_state.clear_input = True
                    st.rerun()
                else:
                    # Get AI response for other requests
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
                            st.session_state.chat_history.append({
                                'role': 'assistant',
                                'content': f"Error getting AI response: {e}",
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
                            st.success(f"✅ Resume '{original_name}' uploaded and processed successfully!")
                            st.session_state.chat_history.append({
                                'role': 'system',
                                'content': f"✅ Resume '{original_name}' uploaded and processed successfully! Please review your preferences.",
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
