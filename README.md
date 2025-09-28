# AI Job Search Agent

An intelligent, agentic job search system that combines AI-powered job matching, CV customization, and strategic career planning. The agent can browse the web, reflect on information, and make strategic decisions to help you find the perfect job opportunities.

## Features

### 🔍 Intelligent Job Search
- Searches 100+ job openings across multiple job sites (LinkedIn, Indeed, Google)
- AI-powered job matching using resume and preferences
- Advanced keyword and semantic matching algorithms
- Real-time job market analysis and trends

### 📄 CV Customization
- Creates customized CVs for specific job applications
- Professional PDF resume generation with ATS optimization
- Clean, readable CV display in chat interface
- Automatic keyword optimization for job requirements

### 🌐 Agentic Capabilities
- **Web Browsing**: Agent can search the internet and browse web pages
- **Strategic Reflection**: Analyzes information and makes strategic career decisions
- **Company Research**: Researches companies, roles, and market trends
- **Memory System**: Maintains conversation context across interactions

### 🤖 AI-Powered Intelligence
- Uses OpenAI GPT-4o-mini for intelligent analysis
- Semantic similarity matching for job relevance
- Natural language processing for CV optimization
- Strategic planning and decision-making capabilities

## Installation

1. **Clone or download** the project to your local machine

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Set up environment variables**:
```bash
# Create .env file with your OpenAI API key
OPENAI_API_KEY=your_api_key_here
```

4. **Prepare your resume**:
   - Place your resume PDF in the `data/` folder as `resume.pdf`
   - The agent will automatically process and extract your resume data

5. **Configure preferences** (optional):
   - The agent will create `data/preferences.json` automatically
   - You can modify job roles, locations, and companies through the UI

## Usage

### Quick Start
```bash
python run_agent.py
```

This will launch the Streamlit web interface in your browser.

### Workflow Steps

1. **Resume Check**: Upload and process your resume
2. **Preferences**: Review and modify job search preferences
3. **Search Prompt**: Customize the job search strategy
4. **Job Search**: Execute AI-powered job search
5. **Results**: View matched job opportunities
6. **Chat & Customize**: Interact with the agent for CV customization and strategic planning

### Chat Capabilities

The agent can help you with:

- **Job Analysis**: "Analyze this job posting for me"
- **CV Customization**: "Create a customized CV for this data scientist role at Google"
- **Company Research**: "Research this company's culture and recent news"
- **Strategic Planning**: "What should I focus on to improve my chances for this role?"
- **Market Analysis**: "What are the current trends in AI engineering roles?"
- **Salary Research**: "What's the typical salary range for this position?"

### Example Interactions

```
You: "Create a customized CV for the AI Engineer role at Microsoft"
Agent: [Creates customized CV, displays content, provides download links]

You: "Research Microsoft's AI initiatives and company culture"
Agent: [Browses web, analyzes information, provides strategic insights]

You: "What changes did you make to my CV for this role?"
Agent: [Explains specific modifications and reasoning]
```

## File Structure

```
Job_search_agent/
├── src/
│   ├── agent_core.py            # Core AI agent with all tools
│   └── streamlit_ui.py          # Streamlit user interface
├── data/
│   ├── resume.pdf               # Your resume PDF
│   ├── preferences.json         # Job preferences
│   ├── base_cv.json            # Extracted resume data
│   ├── job_results.json        # Top job matches
│   └── CV_*.json               # Customized CVs
├── run_agent.py                # Main launcher script
├── requirements.txt            # Python dependencies
├── WORKFLOW_V6.md              # Detailed workflow documentation
└── README.md                   # This file
```

## Key Features

1. **Agentic Intelligence**: Web browsing and strategic reflection capabilities
2. **Streamlined Workflow**: Clean 6-step process with visual progress indicators
3. **Professional CV Generation**: High-quality PDF output with ATS optimization
4. **Real-time Display**: Customized CVs shown in chat interface for immediate review
5. **Memory System**: Maintains conversation context across interactions
6. **Flexible Preferences**: Full control over job roles, locations, and companies
7. **Strategic Planning**: Agent can research companies and provide career advice
8. **Clean Architecture**: Separated core logic from user interface

## Dependencies

### Core AI & Agent Framework
- `langgraph` - Agent framework for tool orchestration
- `langchain` - LLM integration and tool management
- `langchain-openai` - OpenAI integration
- `openai` - Direct OpenAI API access

### Job Search & Web Scraping
- `jobspy` - Job scraping from multiple platforms
- `requests` - HTTP requests for web browsing
- `beautifulsoup4` - HTML parsing for web content

### PDF Processing & Generation
- `pdfplumber` - PDF text extraction
- `reportlab` - Professional PDF generation

### Data Processing & ML
- `scikit-learn` - Similarity matching algorithms
- `pandas` - Data manipulation
- `numpy` - Numerical operations

### User Interface
- `streamlit` - Web application framework

### Configuration
- `python-dotenv` - Environment variable management

## Technical Notes

- **Resume Processing**: Automatically extracts and structures resume data from PDF
- **Memory Management**: Maintains conversation context using LangGraph's memory system
- **Tool Integration**: Seamless integration of web browsing, reflection, and CV customization tools
- **Error Handling**: Robust error handling with fallback mechanisms
- **Performance**: Optimized for speed with caching and efficient algorithms

## Support

For questions or issues, please refer to the `WORKFLOW.md` file for detailed workflow documentation.