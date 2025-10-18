# AI Job Search Agent - Comprehensive Workflow Documentation

## Overview
The AI Job Search Agent is an intelligent, autonomous system that combines advanced job searching, AI-powered CV customization, and strategic career planning. It features sophisticated mechanisms for job discovery, intelligent ranking, memory management, and follow-up interactions with full customization capabilities.

## Architecture & Core Mechanisms

### System Architecture
- **`agent_core.py`**: Core AI agent with LangGraph framework, tools, and intelligent workflows
- **`streamlit_ui.py`**: Streamlit web interface for user interaction
- **`run_agent.py`**: Main launcher script with chatbot mode
- **Memory System**: Intelligent conversation memory with persistence and token management

### AI Agent Framework
- **Framework**: LangGraph with ReAct (Reasoning + Acting) pattern
- **Model**: OpenAI GPT-4o-mini with temperature=0.1 for consistent, factual responses
- **Memory**: Persistent chat memory with intelligent cleanup and context preservation
- **Tools**: 15+ specialized tools for job search, CV customization, and strategic analysis

## Job Search Mechanisms

### 1. Multi-Platform Job Discovery
**Tool**: `jobspy_search`

**Process**:
- **Platforms**: LinkedIn, Indeed, Google Jobs (configurable)
- **Search Strategy**: Multiple requests with offsets to gather 200+ jobs
- **Rate Limiting**: 1-second delays between requests for respectful scraping
- **Deduplication**: Removes duplicate jobs based on title, company, and location
- **Data Extraction**: Title, company, location, description, URL, date posted

**Technical Details**:
```python
# Example search parameters
search_term = "AI Engineer"
location = "London, UK"
results_wanted = 200
hours_old = 72  # Only recent postings
sites = ["linkedin", "indeed", "google"]
```

### 2. Intelligent Job Ranking System
**Tool**: `find_best_job_matches`

**Two-Stage Ranking Process**:

#### Stage 1: Semantic Similarity Matching
- **Resume Embedding**: Converts resume text to vector using OpenAI text-embedding-3-small
- **Preferences Embedding**: Converts job preferences to vector
- **Job Embedding**: Converts job title + company + location + description to vector
- **Similarity Calculation**: Cosine similarity between vectors
- **Weighted Score**: 70% resume similarity + 30% preferences similarity

#### Stage 2: Keyword-Based Reranking
- **Keyword Extraction**: LLM analyzes preferred roles to extract relevant keywords
- **Categories**: Core technology, leadership level, domain, industry, skills
- **Enhanced Weights**: Leadership level (50%), Core technology (30%), Domain (15%)
- **Leadership Boost**: 50% score boost for exact leadership term matches
- **Dynamic Scoring**: Leadership roles get 60% similarity + 40% keywords vs 80% + 20% for others

**Scoring Formula**:
```python
# For leadership roles (Head, Director, Lead, Manager, etc.)
final_score = 0.6 * similarity_score + 0.4 * keyword_score

# For other roles
final_score = 0.8 * similarity_score + 0.2 * keyword_score
```

### 3. Advanced Filtering System
**Negative Keyword Filtering**:
- **Enhanced Keywords**: intern, internship, trainee, graduate, fresher, entry-level, junior, assistant
- **Multi-Field Check**: Searches title, company, AND description
- **Comprehensive Coverage**: Includes variations like "part-time", "contract", "freelance"
- **Logging**: Shows which jobs were excluded and why

**Example Filtering**:
```
🚫 Excluded: Junior Data Scientist at TechCorp (matched: junior)
🚫 Excluded: AI Internship at StartupXYZ (matched: internship)
🔍 Filtered out 15 jobs with negative keywords
```

## CV Customization Mechanisms

### 1. Complete CV Workflow
**Tool**: `create_customized_resume`

**Process**:
1. **Job Description Fetching**: Retrieves full job description from URL
2. **Requirements Analysis**: LLM extracts technical skills, soft skills, experience level
3. **CV Customization**: Tailors existing CV content to match job requirements
4. **JSON Generation**: Creates structured JSON file with metadata
5. **PDF Generation**: Generates professional PDF using Harvard template
6. **Memory Storage**: Stores CV data for future reference

### 2. CV Customization Algorithm
**Tool**: `customize_cv`

**Preservation Principles**:
- **No Fabrication**: Never creates false information
- **Minimal Changes**: Only reorders and rephrases existing content
- **Keyword Integration**: Uses job-relevant keywords in existing descriptions
- **Structure Preservation**: Maintains exact JSON structure
- **Truthfulness**: All content remains accurate and verifiable

**Customization Process**:
1. **Keyword Extraction**: Identifies 12 high-priority keywords from job description
2. **Content Matching**: Tags relevant experiences and skills
3. **Achievement Focus**: Converts responsibilities to achievement-focused bullets
4. **Reordering**: Moves most relevant bullets to top (max 2-3 per role)
5. **Rephrasing**: Uses job terminology while maintaining truthfulness

### 3. Professional PDF Generation
**Tool**: `generate_cv_pdf`

**Harvard Template Features**:
- **Clean Layout**: Professional formatting with proper hierarchy
- **Compact Spacing**: Optimized for 1-2 pages (reduced from 3 pages)
- **ATS Optimization**: Clean formatting for Applicant Tracking Systems
- **Style Management**: Unique CV-specific styles to prevent conflicts
- **Error Handling**: Robust HTML escaping and date formatting

**Spacing Optimization**:
```python
# Reduced spacing for compact layout
CVSectionHeader: spaceBefore=6, spaceAfter=3, leading=12
CVJobTitle: spaceBefore=3, spaceAfter=1
CVBullet: spaceAfter=1, leading=11
CVBodyText: spaceAfter=2, leading=11
```

## Memory & Follow-up Mechanisms

### 1. Intelligent Memory System
**Features**:
- **Persistent Storage**: Maintains context across sessions
- **Smart Cleanup**: Preserves important data while managing token limits
- **Data Categorization**: Separates job results, CV data, and conversation
- **Token Awareness**: Prevents context overflow with intelligent pruning

**Memory Types**:
- **Job Results**: Complete job search data with scores and analysis
- **CV Data**: Customized CV information and file paths
- **Conversation**: Chat history with timestamps
- **System Context**: Resume info and preferences

### 2. Follow-up Question Capabilities
**Memory Integration**:
- **Job Context**: "Show me the jobs again", "What was the top match?"
- **CV Context**: "Compare my CVs", "Explain the modifications"
- **Strategic Analysis**: "What should I focus on?", "Research this company"

**Example Interactions**:
```
User: "Create a CV for job #3"
Agent: [Creates customized CV, stores in memory]

User: "What changes did you make?"
Agent: [Retrieves CV data from memory, explains modifications]

User: "Show me the jobs again"
Agent: [Retrieves job results from memory, displays formatted list]
```

### 3. Advanced Search Capabilities
**Web Browsing Tool**: `browse_web`
- **Real-time Information**: Searches current job market data
- **Company Research**: Gathers company culture and recent news
- **Salary Research**: Finds current salary ranges and trends
- **Industry Analysis**: Discovers market trends and opportunities

**Strategic Reflection Tool**: `reflect_on_information`
- **Critical Analysis**: Evaluates information and makes strategic decisions
- **Decision Framework**: Provides structured analysis with recommendations
- **Risk Assessment**: Identifies potential challenges and opportunities

## Workflow Steps

### 1. Resume Processing
- **PDF Extraction**: Uses pdfplumber to extract text from resume.pdf
- **LLM Parsing**: GPT-4o-mini converts unstructured text to structured JSON
- **Data Validation**: Ensures all fields are properly formatted
- **Embedding Creation**: Generates vector embeddings for similarity matching

### 2. Preferences Configuration
- **Default Setup**: Creates preferences.json with leadership roles
- **Customization**: User can modify roles, locations, companies
- **Keyword Generation**: LLM extracts relevant keywords from preferences
- **Negative Keywords**: Identifies unwanted job types for filtering

### 3. Job Search Execution
- **Multi-Site Search**: Searches LinkedIn, Indeed, Google simultaneously
- **Deduplication**: Removes duplicate jobs across platforms
- **Similarity Matching**: Computes semantic similarity scores
- **Keyword Ranking**: Applies keyword-based reranking
- **Filtering**: Removes unwanted job types
- **Result Storage**: Saves top 20 matches with detailed analysis

### 4. Interactive Chat & Customization
- **Memory-Aware**: Maintains context of job search and CV creation
- **Tool Integration**: Seamlessly uses all available tools
- **CV Workflow**: Complete customization process with PDF generation
- **Strategic Analysis**: Web research and strategic reflection
- **Comparison Tools**: Detailed CV analysis and modification explanations

## Technical Implementation

### Key Algorithms

#### Semantic Similarity Matching
```python
def compute_similarity(vec_a, vec_b):
    return cosine_similarity([vec_a], [vec_b])[0][0]

similarity_score = 0.7 * resume_similarity + 0.3 * preferences_similarity
```

#### Enhanced Keyword Scoring
```python
# Leadership roles get priority weighting
weights = {
    "leadership_level": 0.5,  # Highest priority
    "core_technology": 0.3,
    "domain": 0.15,
    "industry": 0.03,
    "skills": 0.02
}

# Dynamic scoring based on role type
if is_leadership_role:
    final_score = 0.6 * similarity + 0.4 * keywords
else:
    final_score = 0.8 * similarity + 0.2 * keywords
```

#### Memory Management
```python
def cleanup_chat_memory():
    # Preserve important data (job results, CV data)
    # Clean regular messages (keep first 10 + last 30)
    # Maintain token limits while preserving context
```

### Performance Optimizations
- **Embedding Caching**: Reuses resume and preference embeddings
- **Batch Processing**: Processes multiple jobs simultaneously
- **Smart Filtering**: Early filtering reduces processing load
- **Memory Cleanup**: Prevents token overflow with intelligent pruning
- **Error Handling**: Robust fallbacks for all operations

## File Structure & Data Flow

```
Job_search_agent/
├── src/
│   ├── agent_core.py              # Core agent with 15+ tools
│   └── streamlit_ui.py            # Web interface
├── data/
│   ├── resume.pdf                 # Input resume
│   ├── base_cv.json              # Extracted resume data
│   ├── preferences.json          # Job preferences
│   ├── job_results_v5.json       # Top job matches
│   ├── all_jobs_unfiltered.json  # Complete job dataset
│   ├── CV_*.json                 # Customized CVs
│   └── Resume_*.pdf              # Generated PDFs
├── run_agent.py                  # Main launcher
└── requirements.txt              # Dependencies
```

## Usage Examples

### Job Search & Analysis
```python
# Agent automatically:
# 1. Searches 200+ jobs across platforms
# 2. Applies semantic similarity matching
# 3. Reranks with keyword analysis
# 4. Filters out unwanted roles
# 5. Returns top 20 matches with scores
```

### CV Customization
```python
# User: "Create a CV for job #3"
# Agent:
# 1. Fetches job description
# 2. Analyzes requirements
# 3. Customizes CV content
# 4. Generates JSON and PDF
# 5. Stores in memory for follow-up
```

### Follow-up Interactions
```python
# User: "What changes did you make?"
# Agent retrieves CV data from memory and explains modifications

# User: "Show me the jobs again"
# Agent retrieves job results from memory and displays formatted list

# User: "Research this company"
# Agent browses web and provides strategic insights
```

## Benefits & Capabilities

### Intelligent Automation
- **AI-Powered Matching**: Semantic similarity + keyword analysis
- **Autonomous Operation**: Agent makes strategic decisions
- **Context Awareness**: Maintains conversation memory
- **Real-time Research**: Web browsing for current information

### Professional Output
- **ATS-Optimized PDFs**: Clean, professional formatting
- **Comprehensive Analysis**: Detailed job and CV analysis
- **Strategic Insights**: Market research and career advice
- **Quality Assurance**: Error handling and validation

### User Experience
- **Streamlined Workflow**: Clear 6-step process
- **Full Control**: Customizable preferences and prompts
- **Transparency**: See exactly what the agent is doing
- **Flexibility**: Modify any aspect of the process

### Technical Excellence
- **Robust Architecture**: LangGraph framework with tool integration
- **Memory Management**: Intelligent token and context management
- **Performance**: Optimized algorithms and caching
- **Reliability**: Comprehensive error handling and fallbacks

This AI Job Search Agent represents a sophisticated integration of modern AI capabilities with practical job search needs, providing users with an intelligent, autonomous system for finding and applying to the perfect job opportunities.