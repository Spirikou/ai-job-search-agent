# AI Job Search Agent - Workflow Documentation

## Overview
The AI Job Search Agent is an intelligent system that combines job searching, CV customization, and strategic career planning. It features a streamlined workflow with full user control over preferences and search parameters, enhanced with agentic capabilities for web browsing and strategic reflection.

## Architecture

### Core Components
- **`agent_core.py`**: Contains the AI agent with all tools and capabilities
- **`streamlit_ui.py`**: Provides the Streamlit user interface
- **`run_agent.py`**: Main launcher script

### Agent Tools & Capabilities

#### Core Job Search Tools
- **`jobspy_search`**: Searches for jobs using the jobspy library
- **`find_best_job_matches`**: Uses AI to find the best job matches based on resume
- **`fetch_job_description`**: Retrieves detailed job descriptions
- **`analyze_job_requirements`**: Analyzes job requirements and matches them to skills

#### CV Customization Tools
- **`customize_cv`**: Customizes CV content for specific job applications
- **`generate_cv_json`**: Creates structured JSON version of customized CV
- **`generate_cv_pdf`**: Generates professional PDF from customized CV
- **`read_resume_file`**: Reads and processes resume files
- **`list_resume_files`**: Lists available resume files
- **`explain_cv_modifications`**: Explains what changes were made to the CV
- **`create_customized_resume`**: Complete CV customization workflow

#### Strategic Agentic Tools
- **`browse_web`**: Searches the internet and browses web pages for information
- **`reflect_on_information`**: Critically analyzes information and makes strategic decisions

## Workflow Steps

### 1. Resume Check
- **Requirement**: Place your resume as `resume.pdf` in the `data/` folder
- **Action**: Click "Check for Resume" to load and convert your resume
- **Next**: Automatically proceeds to preferences step

### 2. Preferences Review & Modification
- **Display**: Shows current preferences from `data/preferences.json`
- **Sections**:
  - Job Roles (one per line)
  - Locations (one per line) 
  - Companies (one per line)
- **Action**: Modify and save preferences
- **Next**: Proceeds to prompt step

### 3. Job Search Prompt Review & Modification
- **Display**: Shows the current job search prompt
- **Action**: Review and modify the prompt as needed
- **Default**: Includes AI job search with similarity matching
- **Next**: Proceeds to search step

### 4. Job Search Execution
- **Display**: Shows search parameters summary
- **Action**: Click "Start Job Search" to execute
- **Process**: Uses the AI agent with your prompt and preferences
- **Next**: Proceeds to results step

### 5. Results View
- **Display**: Shows job search results (if structured data available)
- **Fallback**: Shows chat interface with search results
- **Action**: Proceed to chat for detailed analysis

### 6. Chat & Customize
- **Features**:
  - Ask questions about jobs
  - Request CV customizations
  - Get career advice
  - Analyze job requirements
  - Research companies and roles using web browsing
  - Strategic reflection on career decisions
- **CV Creation**: Use chat to request customized CVs for specific jobs
- **Agentic Capabilities**: The agent can browse the web, reflect on information, and make strategic decisions

## Key Features

1. **Intelligent Job Search**: AI-powered job matching with similarity analysis
2. **CV Customization**: Automatic CV tailoring for specific job applications
3. **Web Browsing**: Agent can search the internet and browse web pages
4. **Strategic Reflection**: Agent can analyze information and make strategic decisions
5. **Preference Management**: Full control over job roles, locations, and companies
6. **Prompt Customization**: Ability to modify the job search prompt
7. **Structured Workflow**: Clear 6-step process with status indicators
8. **Enhanced Chat**: Better integration with job search results
9. **PDF Generation**: Professional PDF output for customized CVs
10. **Memory System**: Maintains conversation context across interactions

## File Structure

```
Job_search_agent/
├── data/
│   ├── resume.pdf          # Your resume (required)
│   └── preferences.json    # Your preferences (auto-created if missing)
├── src/
│   ├── agent_core.py      # Core AI agent with all tools
│   └── streamlit_ui.py    # Streamlit user interface
├── run_agent.py           # Main launcher script
├── requirements.txt       # Python dependencies
└── README.md              # Documentation
```

## Usage

1. Install dependencies: `pip install -r requirements.txt`
2. Set up environment variables (OpenAI API key)
3. Place your resume as `resume.pdf` in the `data/` folder
4. Run: `python run_agent.py`
5. Follow the 6-step workflow
6. Use chat for CV customization, web research, and strategic planning

## Benefits

- **Intelligent Automation**: AI-powered job matching and CV customization
- **Strategic Planning**: Agent can research companies, analyze market trends, and provide strategic career advice
- **User Control**: Full control over preferences and search parameters
- **Transparency**: See exactly what search is being performed
- **Flexibility**: Modify prompts and preferences as needed
- **Efficiency**: Streamlined workflow with clear progress indicators
- **Integration**: Seamless chat interface for CV customization and strategic planning
- **Professional Output**: High-quality PDF generation for customized CVs
- **Memory**: Maintains context across conversations for better assistance

