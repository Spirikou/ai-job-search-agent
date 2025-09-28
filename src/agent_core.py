from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.tools import tool
from typing import Optional, List, Dict, Any
from jobspy import scrape_jobs
from dotenv import load_dotenv
import json
import re
import time
import threading
import requests
import os
from datetime import datetime, date
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
import pdfplumber
from openai import OpenAI

# Load environment variables from .env file
load_dotenv()

def serialize_for_json(obj):
    """Convert objects to JSON-serializable format."""
    if isinstance(obj, (date, datetime)):
        return obj.strftime('%Y-%m-%d')
    elif hasattr(obj, 'isoformat'):
        return obj.isoformat()
    elif hasattr(obj, '__dict__'):
        return {k: serialize_for_json(v) for k, v in obj.__dict__.items()}
    elif isinstance(obj, dict):
        return {k: serialize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [serialize_for_json(item) for item in obj]
    else:
        return obj

# ============================================================================
# UTILITY CLASSES AND FUNCTIONS
# ============================================================================

def get_llm_model():
    """Get a configured LLM model instance with enhanced memory settings."""
    return ChatOpenAI(
        model="gpt-4o-mini",
        max_tokens=4000,  # Increased from default for longer responses
        temperature=0.7,   # Balanced creativity and consistency
        request_timeout=60  # Allow more time for complex operations
    )

def clean_json_response(content: str) -> str:
    """Clean JSON response by removing markdown formatting."""
    content = content.strip()
    if content.startswith('```json'):
        content = content[7:]
    if content.endswith('```'):
        content = content[:-3]
    return content.strip()

def parse_json_response(content: str, error_context: str = "JSON parsing") -> dict:
    """Parse JSON response with error handling."""
    try:
        cleaned_content = clean_json_response(content)
        return json.loads(cleaned_content)
    except json.JSONDecodeError as e:
        print(f"❌ Error in {error_context}: {e}")
        print(f"❌ Problematic content: {content[:200]}...")
        raise e

def get_data_dir() -> Path:
    """Get the data directory path."""
    return Path(__file__).parent.parent / "data"

class ProgressIndicator:
    """Visual progress indicator for long-running operations."""
    
    def __init__(self, message="Working"):
        self.message = message
        self.running = False
        self.thread = None
    
    def start(self):
        """Start the progress indicator."""
        self.running = True
        self.thread = threading.Thread(target=self._animate)
        self.thread.daemon = True
        self.thread.start()
    
    def stop(self):
        """Stop the progress indicator."""
        self.running = False
        if self.thread:
            self.thread.join()
        # Clear the entire line and move cursor to beginning
        print("\r" + " " * 80 + "\r", end="", flush=True)
    
    def _animate(self):
        """Animation loop."""
        chars = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
        i = 0
        while self.running:
            print(f"\r{chars[i % len(chars)]} {self.message}...", end="", flush=True)
            i += 1
            time.sleep(0.1)



def extract_keywords_from_roles(preferred_roles: List[str], llm_model: ChatOpenAI) -> Dict[str, List[str]]:
    """
    Use LLM to extract relevant keywords from preferred roles for job title matching.
    
    Args:
        preferred_roles: List of preferred job roles
        llm_model: LLM model for keyword extraction
    
    Returns:
        Dictionary with different keyword categories
    """
    roles_text = ", ".join(preferred_roles)
    
    prompt = f"""
    Analyze these job roles and extract keywords that would appear in job titles for similar positions:
    
    Preferred Roles: {roles_text}
    
    Extract keywords in these categories:
    1. Core Technology Keywords (AI, ML, Data, etc.)
    2. Leadership/Level Keywords (Lead, Head, Director, etc.)
    3. Domain Keywords (Engineering, Product, Strategy, etc.)
    4. Industry Keywords (Tech, Finance, Healthcare, etc.)
    5. Skill Keywords (Python, AWS, etc.)
    6. Negative Keywords (words that indicate unwanted job types based on the preferred roles)
    
    Return ONLY valid JSON in this format:
    {{
        "core_technology": ["keyword1", "keyword2"],
        "leadership_level": ["keyword1", "keyword2"],
        "domain": ["keyword1", "keyword2"],
        "industry": ["keyword1", "keyword2"],
        "skills": ["keyword1", "keyword2"],
        "alternative_titles": ["alternative title 1", "alternative title 2"],
        "negative_keywords": ["keyword1", "keyword2"]
    }}
    
    For negative keywords, analyze the preferred roles and determine what job types would be UNWANTED:
    - If looking for leadership roles (Head, Director, Lead), exclude: intern, junior, entry, trainee, apprentice, assistant, coordinator
    - If looking for technical roles, exclude: sales, retail, customer service, call center, marketing
    - If looking for full-time roles, exclude: part-time, temp, contract, freelance, seasonal
    - If looking for senior roles, exclude: junior, entry, graduate, trainee, intern
    
    Focus on keywords that would appear in job titles, not job descriptions.
    Include variations and synonyms.
    """
    
    response = llm_model.invoke([
        {"role": "system", "content": "You are an expert at analyzing job roles and extracting relevant keywords for job matching."},
        {"role": "user", "content": prompt}
    ])
    
    try:
        keywords = parse_json_response(response.content, "keyword extraction")
        return keywords
    except Exception as e:
        print(f"❌ Error extracting keywords: {e}")
        raise e

def compute_keyword_score(job_title: str, keywords: Dict[str, List[str]]) -> float:
    """
    Compute a keyword-based score for job title matching.
    
    Args:
        job_title: Job title to score
        keywords: Dictionary of keyword categories (excluding negative_keywords)
    
    Returns:
        Score between 0 and 1
    """
    title_lower = job_title.lower()
    total_score = 0.0
    total_weight = 0.0
    
    # Weight different keyword categories
    weights = {
        "core_technology": 0.4,  # Most important
        "leadership_level": 0.3,
        "domain": 0.2,
        "industry": 0.05,
        "skills": 0.05
    }
    
    # Calculate positive keyword scores (excluding negative_keywords)
    for category, keyword_list in keywords.items():
        if category in weights and category != "negative_keywords":
            matches = sum(1 for keyword in keyword_list if keyword.lower() in title_lower)
            if keyword_list:  # Avoid division by zero
                category_score = matches / len(keyword_list)
                total_score += category_score * weights[category]
                total_weight += weights[category]
    
    # Return score between 0 and 1
    return total_score / total_weight if total_weight > 0 else 0.0

def save_job_results(results: List[Dict[str, Any]], filename: str = "job_results_v5.json") -> str:
    """Save job results to a JSON file in the data folder."""
    data_dir = get_data_dir()
    data_dir.mkdir(exist_ok=True)
    
    file_path = data_dir / filename
    
    enhanced_results = {
        "metadata": {
            "total_jobs": len(results),
            "generated_at": datetime.now().isoformat(),
            "search_criteria": "Job search with CV customization"
        },
        "jobs": results
    }
    
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(enhanced_results, f, indent=2, ensure_ascii=False)
    
    return str(file_path)

def save_all_jobs(all_jobs: List[Dict[str, Any]], filename: str = "all_jobs_unfiltered.json") -> str:
    """Save all unfiltered jobs to a JSON file for performance analysis."""
    data_dir = get_data_dir()
    data_dir.mkdir(exist_ok=True)
    
    file_path = data_dir / filename
    
    enhanced_results = {
        "metadata": {
            "total_jobs": len(all_jobs),
            "generated_at": datetime.now().isoformat(),
            "search_criteria": "All jobs found before filtering and ranking",
            "note": "This file contains all jobs found before any filtering or ranking for performance analysis"
        },
        "jobs": all_jobs
    }
    
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(enhanced_results, f, indent=2, ensure_ascii=False)
    
    return str(file_path)

def store_job_results_in_memory(job_results: List[Dict[str, Any]]):
    """Store job results in chat memory for dynamic querying."""
    job_summary = {
        "type": "job_results",
        "total_jobs": len(job_results),
        "jobs": job_results[:20],  # Store top 20 for context
        "timestamp": datetime.now().isoformat()
    }
    
    chat_memory.append({
        "role": "system", 
        "content": f"Job search results stored: {len(job_results)} jobs found. Top matches include: " + 
                  ", ".join([f"{job.get('title', 'N/A')} at {job.get('company', 'N/A')}" for job in job_results[:5]])
    })
    
    chat_memory.append({
        "role": "system",
        "content": f"JOB_RESULTS_DATA: {json.dumps(job_summary, indent=2)}"
    })


# ============================================================================
# AGENT TOOLS
# ============================================================================
@tool("jobspy_search", return_direct=False)
def jobspy_search(
    search_term: str,
    location: str = "London, UK",
    site_name: Optional[List[str]] = None,
    results_wanted: int = 200,  # Increased to 200
    hours_old: int = 72,
    country_indeed: str = "UK"
) -> List[dict]:
    """
    Search for job openings across multiple job sites (LinkedIn, Indeed, Google, etc.).
    This tool searches each site separately with multiple requests using offsets to get ~200 jobs.
    
    Args:
        search_term: Job title or keywords to search for (e.g., 'AI engineer', 'data scientist').
        location: Where to search for jobs (e.g., 'London, UK', 'New York, NY').
        site_name: List of job sites to search (indeed, linkedin, google, glassdoor, etc.).
        results_wanted: Target number of jobs to fetch (default: 200).
        hours_old: Only return jobs posted in last X hours (default: 72).
        country_indeed: Which Indeed country site to use (default: 'UK').
    
    Returns:
        A list of unique job dictionaries with title, company, location, description, and URL.
    """
    sites = site_name or ["linkedin", "indeed", "google"]
    print(f"🔍 Searching for ~{results_wanted} jobs...")
    
    all_jobs = []
    
    try:
        # Search each site separately with multiple requests
        for site in sites:
            site_jobs = []
            jobs_per_request = 40  # Maximum per request
            requests_needed = (results_wanted + jobs_per_request - 1) // jobs_per_request  # Ceiling division
            
            for request_num in range(requests_needed):
                try:
                    # Calculate offset for this request
                    offset = request_num * jobs_per_request
                    current_batch_size = min(jobs_per_request, results_wanted - offset)
                    
                    if current_batch_size <= 0:
                        break
                    
                    jobs = scrape_jobs(
                        site_name=[site], 
                        search_term=search_term,
                        location=location,
                        results_wanted=current_batch_size,
                        hours_old=hours_old,
                        country_indeed=country_indeed,
                    )
                    
                    batch_jobs = jobs.to_dict(orient="records")
                    
                    # Ensure all fields are JSON-serializable
                    for job in batch_jobs:
                        # Ensure date_posted field is included
                        if 'date_posted' not in job or job['date_posted'] is None:
                            job['date_posted'] = 'N/A'
                        
                        # Serialize all fields to handle any date objects
                        job = serialize_for_json(job)
                    
                    site_jobs.extend(batch_jobs)
                    
                    # Small delay between requests to be respectful
                    time.sleep(1)
                    
                except Exception as batch_error:
                    continue
            
            # Remove duplicates from this site's results
            site_unique_jobs = remove_duplicate_jobs(site_jobs)
            all_jobs.extend(site_unique_jobs)
            
            # If we have enough jobs, we can stop early
            if len(all_jobs) >= results_wanted:
                break
        
        # Remove duplicates across all sites
        unique_jobs = remove_duplicate_jobs(all_jobs)
        print(f"📋 Found {len(unique_jobs)} jobs")
        
        return unique_jobs
        
    except Exception as e:
        print(f"❌ Search error: {e}")
        return []

@tool("find_best_job_matches", return_direct=False)
def find_best_job_matches(
    jobs: List[dict], 
    top_n: int = 20,  # Changed to 20 as requested
    resume_text: Optional[str] = None,
    preferences_text: Optional[str] = None,
    keyword_threshold: float = 0.3,
    save_results: bool = True
) -> List[dict]:
    """
    Find the best job matches using a two-stage approach: similarity matching + keyword reranking.
    This tool prioritizes semantic similarity and uses keywords as a secondary boost.
    
    Args:
        jobs: List of job dictionaries to match against.
        top_n: Number of top matches to return (default: 20).
        resume_text: Resume text for matching (if None, uses default resume).
        preferences_text: Preferences text for matching (if None, uses default preferences).
        keyword_threshold: Minimum keyword score threshold for first ranking (default: 0.3).
        save_results: Whether to save results to JSON file (default: True).
    
    Returns:
        List of top job matches with similarity scores and detailed analysis.
    """
    # Remove duplicates before processing
    jobs = remove_duplicate_jobs(jobs)
    print(f"🧠 Analyzing {len(jobs)} jobs...")
    
    # Save all unfiltered jobs for performance analysis
    save_all_jobs(jobs)
    
    # Debug: Check if we have the required data
    if not jobs:
        print("❌ No jobs to analyze")
        return []
    
    # Use default resume and preferences if not provided
    if resume_text is None:
        # Load resume data if not already loaded
        if resume is None:
            try:
                resume_data = json.load(open("data/base_cv.json"))
                resume_text = build_resume_text(resume_data)
            except Exception as e:
                print(f"❌ Error loading resume: {e}")
                return []
        else:
            resume_text = build_resume_text(resume)
    
    if preferences_text is None:
        preferences_text = pref_text
    
    # Load preferred roles from preferences
    try:
        preferred_roles = preferences[0].get("roles", [])
    except Exception as e:
        print(f"❌ Error loading preferences: {e}")
        return []
    
    # Initialize models
    llm_model = get_llm_model()
    
    # Generate keywords including negative keywords
    keywords = extract_keywords_from_roles(preferred_roles, llm_model)
    negative_keywords = keywords.get("negative_keywords", [])
    
    # Filter out jobs with negative keywords
    if negative_keywords:
        jobs = filter_jobs_with_negative_keywords(jobs, negative_keywords)
        print(f"🔍 Filtered to {len(jobs)} relevant jobs")
    
    # Use global embeddings and vectors if available, otherwise create new ones
    if resume_vec is None or pref_vec is None:
        resume_vec_local = embeddings.embed_query(resume_text)
        pref_vec_local = embeddings.embed_query(preferences_text)
    else:
        resume_vec_local = resume_vec
        pref_vec_local = pref_vec
    
    def compute_similarity(vec_a, vec_b) -> float:
        """Compute cosine similarity between two embedding vectors."""
        return float(cosine_similarity([vec_a], [vec_b])[0][0])
    
    # Compute similarities for all jobs
    similarity_results = []
    for job in jobs:
        # Use only essential job info to reduce context length
        job_text = f"{job.get('title','')} {job.get('company','')} {job.get('location','')}"
        # Truncate description to prevent context overflow
        description = job.get('description', '')
        if description and len(description) > 200:
            description = description[:200] + "..."
        job_text += f" {description}"
        job_vec = embeddings.embed_query(job_text)
        
        # Compute similarities
        sim_resume = compute_similarity(resume_vec_local, job_vec)
        sim_pref = compute_similarity(pref_vec_local, job_vec)
        
        # Primary similarity score (70% resume + 30% preferences)
        similarity_score = 0.7 * sim_resume + 0.3 * sim_pref
        
        job['similarity_score'] = similarity_score
        job['resume_similarity'] = sim_resume
        job['preferences_similarity'] = sim_pref
        similarity_results.append(job)
    
    # Sort by similarity score (primary ranking)
    similarity_results.sort(key=lambda x: x['similarity_score'], reverse=True)
    
    # Stage 2: Keyword reranking (secondary boost)
    top_similarity_jobs = similarity_results[:min(top_n * 2, 30)]  # Limit to max 30 jobs to prevent context overflow
    
    results = []
    for job in top_similarity_jobs:
        job_title = job.get('title', '')
        keyword_score = compute_keyword_score(job_title, keywords)
        
        # Apply keyword boost to similarity score
        # 80% similarity + 20% keyword boost
        final_score = 0.8 * job['similarity_score'] + 0.2 * keyword_score
        
        # Enhanced result with publishing date and description
        result = {
            "title": job.get("title", "N/A"),
            "company": job.get("company", "N/A"),
            "location": job.get("location", "N/A"),
            "url": job.get("job_url", "N/A"),
            "date_posted": job.get("date_posted", "N/A"),  # Include publishing date
            "description": job.get("description", "N/A"),  # Include job description
            "score": round(final_score, 3),
            "resume_similarity": round(job['resume_similarity'], 3),
            "preferences_similarity": round(job['preferences_similarity'], 3),
            "keyword_score": round(keyword_score, 3),
            "matching_keywords": extract_matching_keywords(job_title, keywords)[:3]  # Limit to 3 keywords
        }
        results.append(result)
    
    # Sort by final score (similarity + keyword boost)
    results.sort(key=lambda x: x["score"], reverse=True)
    top_results = results[:top_n]
    
    if save_results:
        save_job_results(top_results)
    
    return top_results

@tool("browse_web", return_direct=False)
def browse_web(url: str, search_query: str = None) -> str:
    """
    Browse the web to fetch content from a URL or search for information.
    This tool enables the agent to gather real-time information from the internet.
    
    Args:
        url: URL to browse or "search" to perform a web search
        search_query: Query to search for if url is "search"
    
    Returns:
        Content from the webpage or search results
    """
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        if url == "search" and search_query:
            # Perform a web search using DuckDuckGo
            search_url = f"https://duckduckgo.com/html/?q={search_query.replace(' ', '+')}"
            response = requests.get(search_url, headers=headers, timeout=15)
            response.raise_for_status()
            
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract search results
            results = []
            result_elements = soup.find_all('div', class_='result')[:10]  # Top 10 results
            
            for result in result_elements:
                title_elem = result.find('a', class_='result__a')
                snippet_elem = result.find('div', class_='result__snippet')
                
                if title_elem and snippet_elem:
                    title = title_elem.get_text(strip=True)
                    url_found = title_elem.get('href', '')
                    snippet = snippet_elem.get_text(strip=True)
                    
                    results.append(f"**{title}**\nURL: {url_found}\nSummary: {snippet}\n")
            
            return f"Search results for '{search_query}':\n\n" + "\n".join(results[:5])  # Top 5 results
        
        else:
            # Browse specific URL
            response = requests.get(url, headers=headers, timeout=15)
            response.raise_for_status()
            
            # Basic HTML parsing to extract text content
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style", "nav", "footer", "aside"]):
                script.decompose()
            
            # Get text content
            text = soup.get_text()
            
            # Clean up the text
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            
            # Limit content length and add source info
            content = text[:8000] if len(text) > 8000 else text
            return f"Content from {url}:\n\n{content}"
        
    except Exception as e:
        return f"Error browsing {url}: {str(e)}"

@tool("reflect_on_information", return_direct=False)
def reflect_on_information(information: str, context: str, decision_needed: str) -> str:
    """
    Reflect on gathered information to make informed decisions and plan next actions.
    This tool enables the agent to think critically about information and plan strategically.
    
    Args:
        information: The information to reflect on
        context: The current context or situation
        decision_needed: What decision or action is needed
    
    Returns:
        Reflection analysis and recommended next steps
    """
    llm_model = get_llm_model()
    
    prompt = f"""
    You are an intelligent AI agent capable of critical thinking and strategic planning.
    
    CONTEXT: {context}
    
    INFORMATION TO REFLECT ON:
    {information}
    
    DECISION NEEDED: {decision_needed}
    
    Please provide a thoughtful reflection that includes:
    
    1. **INFORMATION ANALYSIS**
       - What are the key insights from this information?
       - What patterns or trends do you notice?
       - What information might be missing or needs verification?
    
    2. **STRATEGIC ASSESSMENT**
       - How does this information relate to the current context?
       - What opportunities or challenges does this present?
       - What are the potential risks or benefits?
    
    3. **DECISION FRAMEWORK**
       - What are the possible options or approaches?
       - What criteria should be used to evaluate these options?
       - What additional information might be needed?
    
    4. **RECOMMENDED ACTIONS**
       - What is the best course of action based on this analysis?
       - What specific next steps should be taken?
       - What should be monitored or tracked going forward?
    
    5. **CONFIDENCE ASSESSMENT**
       - How confident are you in this analysis? (High/Medium/Low)
       - What assumptions are you making?
       - What could change your recommendation?
    
    Provide a clear, actionable reflection that demonstrates critical thinking and strategic planning.
    """
    
    response = llm_model.invoke([
        {"role": "system", "content": "You are an expert strategic analyst and decision-making consultant who provides thoughtful, well-reasoned analysis and recommendations."},
        {"role": "user", "content": prompt}
    ])
    
    return response.content.strip()

@tool("plan_next_actions", return_direct=False)
def plan_next_actions(current_situation: str, goals: str, available_tools: str) -> str:
    """
    Plan the next sequence of actions based on the current situation and goals.
    This tool enables the agent to think ahead and create strategic action plans.
    
    Args:
        current_situation: Description of the current state
        goals: What the agent is trying to achieve
        available_tools: List of tools available to the agent
    
    Returns:
        Strategic action plan with specific steps
    """
    llm_model = get_llm_model()
    
    prompt = f"""
    You are an AI agent planning your next actions strategically.
    
    CURRENT SITUATION: {current_situation}
    
    GOALS: {goals}
    
    AVAILABLE TOOLS: {available_tools}
    
    Create a strategic action plan that includes:
    
    1. **PRIORITY ASSESSMENT**
       - What are the most important objectives right now?
       - What are the time-sensitive actions?
       - What dependencies exist between actions?
    
    2. **ACTION SEQUENCE**
       - List 3-5 specific actions in order of priority
       - For each action, specify:
         * What tool to use
         * What parameters or inputs needed
         * Expected outcome
         * How it contributes to the goal
    
    3. **CONTINGENCY PLANNING**
       - What could go wrong with each action?
       - What alternative approaches exist?
       - How to adapt if information changes?
    
    4. **SUCCESS METRICS**
       - How will you know if each action succeeded?
       - What information will indicate progress toward goals?
       - When to reassess and potentially change course?
    
    Provide a clear, executable action plan that demonstrates strategic thinking.
    """
    
    response = llm_model.invoke([
        {"role": "system", "content": "You are an expert strategic planner who creates detailed, actionable plans for achieving complex objectives."},
        {"role": "user", "content": prompt}
    ])
    
    return response.content.strip()

def get_formatted_cv_content_for_display(customized_cv: Dict[str, Any], job_title: str, company: str) -> str:
    """Format CV content for display in chat using LLM."""
    try:
        llm_model = get_llm_model()
        
        prompt = f"""
        You are a professional resume formatter. Convert this JSON CV data into a clean, readable format for chat display.
        
        TARGET JOB: {job_title} at {company}
        
        CV DATA (JSON):
        {json.dumps(customized_cv, indent=2)}
        
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
            return build_readable_cv_fallback(customized_cv)
        
        return formatted_content
        
    except Exception as e:
        print(f"❌ Error formatting CV content: {e}")
        # Fallback to basic formatting if LLM fails
        return build_readable_cv_fallback(customized_cv)

def build_readable_cv_fallback(cv_data: Dict[str, Any]) -> str:
    """Build a readable version of the CV from JSON data as fallback."""
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

@tool("research_company_and_role", return_direct=False)
def research_company_and_role(company_name: str, role_title: str, user_goals: str = None) -> str:
    """
    Research a specific company and role to provide strategic insights for job applications.
    This tool demonstrates agentic behavior by autonomously gathering and analyzing information.
    
    Args:
        company_name: Name of the company to research
        role_title: Job title or role to research
        user_goals: User's specific goals or questions (optional)
    
    Returns:
        Comprehensive research report with strategic insights
    """
    try:
        # Step 1: Gather information about the company
        company_search_query = f"{company_name} company information culture values recent news"
        company_info = browse_web("search", company_search_query)
        
        # Step 2: Research the specific role
        role_search_query = f"{role_title} at {company_name} job requirements responsibilities salary"
        role_info = browse_web("search", role_search_query)
        
        # Step 3: Look for recent company news and developments
        news_search_query = f"{company_name} recent news 2024 hiring growth layoffs"
        news_info = browse_web("search", news_search_query)
        
        # Step 4: Reflect on the gathered information
        reflection_context = f"Researching {role_title} position at {company_name} for job application strategy"
        all_info = f"COMPANY INFO:\n{company_info}\n\nROLE INFO:\n{role_info}\n\nRECENT NEWS:\n{news_info}"
        decision_needed = f"How to best position for {role_title} role at {company_name} and what key insights should guide the application strategy"
        
        reflection = reflect_on_information(all_info, reflection_context, decision_needed)
        
        # Step 5: Plan strategic recommendations
        current_situation = f"Researched {company_name} and {role_title} position, gathered market intelligence"
        goals = user_goals or f"Successfully apply for {role_title} position at {company_name}"
        available_tools = "CV customization, job search, web browsing, strategic analysis"
        
        action_plan = plan_next_actions(current_situation, goals, available_tools)
        
        # Step 6: Compile comprehensive report
        report = f"""
# Company & Role Research Report: {role_title} at {company_name}

## Executive Summary
Based on comprehensive research, here are the key insights for your application strategy.

## Company Intelligence
{company_info}

## Role Analysis
{role_info}

## Market Context & Recent Developments
{news_info}

## Strategic Analysis
{reflection}

## Recommended Action Plan
{action_plan}

## Key Takeaways
- **Company Culture**: Focus on values alignment in your application
- **Role Requirements**: Emphasize relevant skills and experience
- **Market Position**: Understand company's current situation and priorities
- **Application Strategy**: Tailor your approach based on gathered insights

This research demonstrates autonomous information gathering and strategic analysis to optimize your job application approach.
"""
        
        return report
        
    except Exception as e:
        return f"Error conducting company research: {str(e)}"

@tool("fetch_job_description", return_direct=False)
def fetch_job_description(job_url: str) -> str:
    """
    Fetch the full job description from a job URL.
    
    Args:
        job_url: URL of the job posting
    
    Returns:
        Full job description text
    """
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        response = requests.get(job_url, headers=headers, timeout=10)
        response.raise_for_status()
        
        # Basic HTML parsing to extract text content
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        
        # Get text content
        text = soup.get_text()
        
        # Clean up the text
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)
        
        return text[:5000]  # Limit to 5000 characters to prevent context overflow
        
    except Exception as e:
        print(f"❌ Error fetching job description: {e}")
        return "Job description not available"

@tool("analyze_job_requirements", return_direct=False)
def analyze_job_requirements(job_description: str, job_title: str, company: str) -> Dict[str, Any]:
    """
    Analyze job requirements and extract key skills, qualifications, and experience needed.
    
    Args:
        job_description: Full job description text
        job_title: Job title
        company: Company name
    
    Returns:
        Dictionary containing analyzed requirements
    """
    llm_model = get_llm_model()
    
    prompt = f"""
    Analyze this job posting and extract the key requirements, skills, and qualifications needed.
    
    Job Title: {job_title}
    Company: {company}
    
    Job Description:
    {job_description}
    
    Extract and organize the following information:
    1. Required technical skills
    2. Required soft skills
    3. Required experience level
    4. Required education/qualifications
    5. Preferred qualifications
    6. Key responsibilities
    7. Industry/domain focus
    8. Company culture indicators
    
    Return ONLY valid JSON in this format:
    {{
        "technical_skills": ["skill1", "skill2"],
        "soft_skills": ["skill1", "skill2"],
        "experience_level": "entry/mid/senior/executive",
        "education": ["requirement1", "requirement2"],
        "preferred_qualifications": ["qual1", "qual2"],
        "key_responsibilities": ["responsibility1", "responsibility2"],
        "industry_focus": "industry name",
        "company_culture": ["culture1", "culture2"],
        "summary": "Brief summary of what this role is looking for"
    }}
    
    Focus on skills and qualifications that would be most important for CV customization.
    """
    
    response = llm_model.invoke([
        {"role": "system", "content": "You are an expert at analyzing job postings and extracting key requirements for CV optimization."},
        {"role": "user", "content": prompt}
    ])
    
    try:
        requirements = parse_json_response(response.content, "job requirements analysis")
        return requirements
        
    except Exception as e:
        print(f"❌ Error analyzing job requirements: {e}")
        return {
            "technical_skills": [],
            "soft_skills": [],
            "experience_level": "mid",
            "education": [],
            "preferred_qualifications": [],
            "key_responsibilities": [],
            "industry_focus": "Technology",
            "company_culture": [],
            "summary": "Job requirements analysis failed"
        }

@tool("customize_cv", return_direct=False)
def customize_cv(job_requirements: Dict[str, Any], job_title: str, company: str) -> Dict[str, Any]:
    """
    Create a customized CV based on job requirements while maintaining truthfulness.
    
    Args:
        job_requirements: Analyzed job requirements
        job_title: Target job title
        company: Target company
    
    Returns:
        Customized CV data structure
    """
    # Load original CV
    try:
        with open("data/base_cv.json", 'r', encoding='utf-8') as f:
            original_cv = json.load(f)
    except Exception as e:
        print(f"❌ Error loading original CV: {e}")
        return {}
    
    llm_model = get_llm_model()
    
    prompt = f"""
    Create a customized version of this CV to better match the job requirements.
    
    TARGET JOB:
    Title: {job_title}
    Company: {company}
    
    JOB REQUIREMENTS:
    {json.dumps(job_requirements, indent=2)}
    
    ORIGINAL CV:
    {json.dumps(original_cv, indent=2)}
    
    CUSTOMIZATION GUIDELINES:
    1. PRESERVE the exact same JSON structure as the original CV
    2. DO NOT add, remove, or fabricate any experiences, skills, or achievements
    3. DO NOT create new content that doesn't exist in the original CV
    4. DO reorder existing content to prioritize job-relevant information
    5. DO rephrase existing descriptions to use job-relevant keywords from the requirements
    6. DO adjust the summary to align with the target role using existing experience
    7. DO reorder skills sections to highlight job-relevant skills first
    8. DO NOT reorder experience section to show most relevant experience first. Keep it in reverse chronological order.
    9. DO emphasize existing achievements that match job requirements
    10. DO use keywords from the job requirements in existing descriptions
    
    MINIMAL CHANGES APPROACH:
    - Keep the CV structure identical
    - Only reorder and rephrase existing content
    - Use job-relevant keywords in existing descriptions
    - Highlight relevant aspects of existing experience
    - Ensure all content remains truthful and accurate
    
    CRITICAL: Return ONLY valid JSON. Do not include any text before or after the JSON. 
    The response must start with {{ and end with }}. Ensure all strings are properly escaped.
    """
    
    response = llm_model.invoke([
        {"role": "system", "content": "You are an expert CV writer who customizes resumes to match job requirements while maintaining complete honesty and accuracy."},
        {"role": "user", "content": prompt}
    ])
    
    try:
        customized_cv = parse_json_response(response.content, "CV customization")
        return customized_cv
        
    except json.JSONDecodeError as e:
        print(f"❌ JSON parsing error: {e}")
        
        # Try to fix common JSON issues
        try:
            content = response.content.strip()
            # Remove any text before the first {
            if '{' in content:
                start_idx = content.find('{')
                content = content[start_idx:]
            
            # Try parsing again
            customized_cv = json.loads(content)
            print("✅ Successfully parsed JSON after cleanup")
            return customized_cv
        except:
            print("❌ Could not fix JSON, returning original CV")
            return original_cv
    except Exception as e:
        print(f"❌ Error customizing CV: {e}")
        return original_cv


@tool("generate_cv_json", return_direct=False)
def generate_cv_json(customized_cv: Dict[str, Any], job_title: str, company: str) -> str:
    """
    Generate a JSON file from the customized CV data.
    
    Args:
        customized_cv: Customized CV data structure
        job_title: Target job title
        company: Target company
    
    Returns:
        Path to generated JSON file
    """
    try:
        # Create filename with job info
        safe_company = re.sub(r'[^\w\s-]', '', company).strip()
        safe_title = re.sub(r'[^\w\s-]', '', job_title).strip()
        filename = f"CV_{safe_company}_{safe_title}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # Create data directory if it doesn't exist
        data_dir = get_data_dir()
        data_dir.mkdir(exist_ok=True)
        
        # Full path to the JSON file
        json_path = data_dir / filename
        
        # Add metadata to the CV
        enhanced_cv = {
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "target_job": job_title,
                "target_company": company,
                "version": "core",
                "description": f"Customized CV for {job_title} at {company}"
            },
            "cv_data": customized_cv
        }
        
        # Save customized CV to JSON file
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(enhanced_cv, f, indent=2, ensure_ascii=False)
        
        return str(json_path)
        
    except Exception as e:
        print(f"❌ Error generating JSON: {e}")
        return ""

@tool("read_resume_file", return_direct=False)
def read_resume_file(file_path: str) -> Dict[str, Any]:
    """Read and return the contents of a resume file (JSON format)."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"❌ Error reading resume file: {e}")
        return {}


@tool("list_resume_files", return_direct=False)
def list_resume_files() -> List[str]:
    """List all resume files in the data directory."""
    try:
        data_dir = get_data_dir()
        return [str(file_path) for file_path in data_dir.glob("*.json") 
                if "cv" in file_path.name.lower() or "resume" in file_path.name.lower()]
    except Exception as e:
        print(f"❌ Error listing resume files: {e}")
        return []

@tool("create_customized_resume", return_direct=False)
def create_customized_resume(job_title: str, company: str, job_url: str = None) -> str:
    """
    Create a complete customized resume workflow for a specific job.
    This tool orchestrates the full CV customization process including PDF generation.
    
    Args:
        job_title: Target job title
        company: Target company
        job_url: Optional job URL to fetch description
    
    Returns:
        Summary of the customization process and file paths
    """
    try:
        print(f"🎯 Starting CV customization for {job_title} at {company}")
        
        # Step 1: Fetch job description if URL provided
        job_description = None
        if job_url:
            print("📥 Fetching job description...")
            job_description = fetch_job_description.invoke({"job_url": job_url})
        
        # Step 2: Analyze job requirements
        print("🧠 Analyzing job requirements...")
        job_requirements = analyze_job_requirements.invoke({
            "job_description": job_description or f"Job title: {job_title} at {company}",
            "job_title": job_title,
            "company": company
        })
        
        # Step 3: Create customized CV
        print("📝 Creating customized CV...")
        customized_cv = customize_cv.invoke({
            "job_requirements": job_requirements,
            "job_title": job_title,
            "company": company
        })
        
        # Check if customization was successful
        if not customized_cv or customized_cv == {}:
            return f"❌ CV customization failed. Please try again or check the job requirements."
        
        # Step 4: Generate JSON file
        print("📄 Generating JSON file...")
        json_path = generate_cv_json.invoke({
            "customized_cv": customized_cv,
            "job_title": job_title,
            "company": company
        })
        
        # Step 5: Generate PDF file
        print("📄 Generating PDF file...")
        pdf_path = generate_cv_pdf.invoke({
            "customized_cv": customized_cv,
            "job_title": job_title,
            "company": company,
            "job_description": job_description
        })
        
        # Check if files were created successfully
        if not pdf_path or not json_path:
            return f"❌ File generation failed. Please try again."
        
        # Step 6: Store in memory
        store_cv_in_memory(customized_cv, job_title, company, pdf_path)
        
        # Step 7: Generate formatted CV content for display
        print("📝 Formatting CV content for display...")
        cv_content = get_formatted_cv_content_for_display(customized_cv, job_title, company)
        
        return f"✅ Customized resume created successfully!\n📄 PDF: {pdf_path}\n📁 JSON: {json_path}\n\n**CV Content:**\n{cv_content}\n\nYou can now use this resume for your application to {job_title} at {company}."
        
    except Exception as e:
        print(f"❌ Error in create_customized_resume: {e}")
        return f"❌ Error creating customized resume: {e}"

@tool("explain_cv_modifications", return_direct=False)
def explain_cv_modifications(customized_cv_path: str, job_title: str = None, company: str = None) -> str:
    """
    Explain the modifications made to a customized CV by comparing it with the original.
    
    Args:
        customized_cv_path: Path to the customized CV JSON file
        job_title: Target job title (optional, will be extracted from CV metadata if available)
        company: Target company (optional, will be extracted from CV metadata if available)
    
    Returns:
        Detailed explanation of CV modifications
    """
    try:
        # Load the customized CV
        with open(customized_cv_path, 'r', encoding='utf-8') as f:
            cv_data = json.load(f)
        
        # Extract CV data and metadata
        if "cv_data" in cv_data:
            customized_cv = cv_data["cv_data"]
            metadata = cv_data.get("metadata", {})
        else:
            customized_cv = cv_data
            metadata = {}
        
        # Get job title and company from metadata or parameters
        target_job_title = job_title or metadata.get("target_job", "Unknown Position")
        target_company = company or metadata.get("target_company", "Unknown Company")
        
        # Load original CV for comparison
        try:
            with open("data/base_cv.json", 'r', encoding='utf-8') as f:
                original_cv = json.load(f)
        except Exception as e:
            return f"❌ Error loading original CV: {e}"
        
        # Create mock job requirements for analysis (since we don't have the original)
        mock_requirements = {
            "technical_skills": ["Extracted from CV analysis"],
            "soft_skills": ["Leadership", "Communication"],
            "experience_level": "mid",
            "education": ["Bachelor's degree or equivalent"],
            "preferred_qualifications": ["Relevant experience"],
            "key_responsibilities": ["Role-specific tasks"],
            "industry_focus": "Technology",
            "company_culture": ["Innovation", "Growth"],
            "summary": f"Requirements for {target_job_title} at {target_company}"
        }
        
        # Analyze modifications
        modification_analysis = analyze_cv_modifications(
            original_cv, 
            customized_cv, 
            target_job_title, 
            target_company, 
            mock_requirements
        )
        
        # Format the explanation for return
        explanation = f"CV Modification Analysis for {target_job_title} at {target_company}\n"
        explanation += "=" * 60 + "\n\n"
        
        overview = modification_analysis.get("overview", {})
        explanation += f"Total Modifications: {overview.get('total_modifications', 0)}\n"
        explanation += f"Sections Modified: {', '.join(overview.get('sections_modified', []))}\n"
        explanation += f"Strategic Focus: {overview.get('strategic_focus', 'N/A')}\n\n"
        
        explanation += "Key Improvements:\n"
        for improvement in overview.get('key_improvements', []):
            explanation += f"• {improvement}\n"
        
        modifications = modification_analysis.get("modifications", [])
        if modifications:
            explanation += "\nDetailed Modifications:\n"
            explanation += "-" * 40 + "\n"
            
            for i, mod in enumerate(modifications, 1):
                explanation += f"\n{i}. {mod.get('section', 'Unknown Section').upper()}\n"
                explanation += f"   Change Type: {mod.get('change_type', 'Unknown').title()}\n"
                explanation += f"   Rationale: {mod.get('rationale', 'N/A')}\n"
                explanation += f"   Job Relevance: {mod.get('job_relevance', 'N/A')}\n"
                explanation += f"   Expected Impact: {mod.get('impact', 'N/A')}\n"
        
        summary = modification_analysis.get("summary", "")
        if summary:
            explanation += f"\nCustomization Strategy:\n{summary}\n"
        
        return explanation
        
    except Exception as e:
        return f"❌ Error analyzing CV modifications: {e}"

def analyze_cv_modifications(original_cv: Dict[str, Any], customized_cv: Dict[str, Any], job_title: str, company: str, job_requirements: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze the differences between original and customized CV and provide detailed explanations.
    
    Args:
        original_cv: Original CV data structure
        customized_cv: Customized CV data structure
        job_title: Target job title
        company: Target company
        job_requirements: Analyzed job requirements
    
    Returns:
        Dictionary containing detailed modification analysis
    """
    llm_model = get_llm_model()
    
    prompt = f"""
    You are an expert CV consultant who analyzes CV customizations and explains the rationale behind each modification.
    
    TARGET JOB:
    Title: {job_title}
    Company: {company}
    
    JOB REQUIREMENTS:
    {json.dumps(job_requirements, indent=2)}
    
    ORIGINAL CV:
    {json.dumps(original_cv, indent=2)}
    
    CUSTOMIZED CV:
    {json.dumps(customized_cv, indent=2)}
    
    Analyze the differences between the original and customized CVs and provide detailed explanations for each modification.
    
    For each section that was modified, explain:
    1. WHAT was changed (specific content, order, emphasis)
    2. WHY it was changed (how it better matches the job requirements)
    3. HOW it improves the CV for this specific role
    4. The strategic reasoning behind the modification
    
    Focus on these key areas:
    - Summary/Objective changes
    - Skills section reordering or emphasis
    - Experience section reordering or rewording
    - Achievement descriptions and quantification
    - Keyword integration from job requirements
    - Overall structure and flow improvements
    
    Return ONLY valid JSON in this format:
    {{
        "overview": {{
            "total_modifications": number,
            "sections_modified": ["section1", "section2"],
            "key_improvements": ["improvement1", "improvement2"],
            "strategic_focus": "brief description of overall strategy"
        }},
        "modifications": [
            {{
                "section": "section_name",
                "change_type": "reorder/rewrite/emphasize/add/remove",
                "original_content": "specific original content",
                "modified_content": "specific modified content",
                "rationale": "detailed explanation of why this change was made",
                "job_relevance": "how this change better matches the job requirements",
                "impact": "expected impact on recruiter/ATS perception"
            }}
        ],
        "keyword_integration": {{
            "job_keywords_added": ["keyword1", "keyword2"],
            "sections_enhanced": ["section1", "section2"],
            "ats_optimization": "explanation of ATS improvements"
        }},
        "summary": "Overall summary of the customization strategy and expected benefits"
    }}
    
    Be specific and detailed in your analysis. Focus on practical, actionable insights that help the user understand the value of each modification.
    """
    
    response = llm_model.invoke([
        {"role": "system", "content": "You are an expert CV consultant who provides detailed analysis of CV customizations and explains the strategic reasoning behind each modification."},
        {"role": "user", "content": prompt}
    ])
    
    try:
        analysis = parse_json_response(response.content, "CV modifications analysis")
        return analysis
        
    except Exception as e:
        print(f"❌ Error analyzing CV modifications: {e}")
        return {
            "overview": {
                "total_modifications": 0,
                "sections_modified": [],
                "key_improvements": [],
                "strategic_focus": "Analysis failed"
            },
            "modifications": [],
            "keyword_integration": {
                "job_keywords_added": [],
                "sections_enhanced": [],
                "ats_optimization": "Analysis failed"
            },
            "summary": "Could not analyze modifications due to technical error"
        }

# Removed display_cv_modification_explanations - now handled by LLM dynamically

@tool("generate_cv_pdf", return_direct=False)
def generate_cv_pdf(customized_cv: Dict[str, Any], job_title: str, company: str, job_description: str = None) -> str:
    """
    Generate a professional PDF resume from customized CV data using the Harvard CV template.
    This tool creates a ready-to-use PDF file that users can download and submit with job applications.
    
    Args:
        customized_cv: Customized CV data structure (from customize_cv tool)
        job_title: Target job title for customization
        company: Target company for customization
        job_description: Target job description for enhanced tailoring (optional)
    
    Returns:
        Path to generated PDF file (e.g., "C:/path/to/Resume_Company_Title_20241201_143022.pdf")
    
    Note: Always use this tool after customize_cv to create the PDF version of the customized resume.
    """
    try:
        # Use the customized CV data directly
        cv_content = customized_cv
        
        llm_model = get_llm_model()
        
        # Prepare job description for the prompt
        job_desc_text = job_description if job_description and job_description != "Job description not available" else None
        
        prompt = f"""
You are a seasoned career consultant and resume expert who produces ATS-friendly, recruiter-focused CVs tailored to a target job posting.

Inputs (provided to you):
- cv_content: a JSON object containing the candidate's existing CV/resume data.
- job_title: the target job title string.
- company: the target company string.
- job_description: the target job description text (if available). If not provided, tailor using job_title and company only.

Hard rules you MUST follow:
1. Do NOT invent facts. Never create company names, job titles, dates, metrics, or certifications that do not appear in cv_content. If a required fact is missing, omit it or use a neutral placeholder like "[not provided]". Do not insert numeric metrics unless the user explicitly provided them.
2. Follow this internal, stepwise process before producing output (apply these steps in order):
   a. Parse the job_description (or job_title/company) and extract up to 12 high-priority keywords, skills, and competencies (technical skills, tools, seniority verbs, metrics, domain words).
   b. Match those keywords to entries in cv_content; tag which experiences, skills, and achievements are relevant.
   c. Convert responsibility statements into achievement-focused bullet points using the formula: Action verb + what you did + result/impact (quantify when the candidate provided numbers).
   d. Prioritize and re-order bullets so that the most relevant achievements for the target job appear first under each role.
3. Use the Harvard CV structure exactly as the layout model (section order and headings), and render section titles in **bold** using Markdown (e.g., **Experience**). The content under each heading must be plain text (no bold or italics).
4. Format details:
   - Contact line under name: Street or city optional; always include email and phone if provided. Use single-line format: Home Street Address • City, State Zip • email • phone
   - Dates: Month Year – Month Year (e.g., April 2020 – August 2023). If only years present, use Year – Year.
   - Bullets: use "• " for each bullet, begin with an action verb, no personal pronouns, phrase-style (not full sentences).
   - Keep the entire CV concise: 1 page for early-career, 2 pages maximum for mid-senior. If cv_content indicates >10 years experience, two pages allowed.
5. Tailoring guidance:
   - Integrate exact job_description keywords into the Skills & Interests and Technical lines when they are present in cv_content.
   - Reword candidate bullets to mirror the job's language (e.g., "built scalable data pipeline" → "designed and deployed scalable data pipelines using X" if candidate used X).
   - Where candidate-provided metrics exist, preserve them and highlight them early in the bullet.
6. Preservation & minimal editing:
   - Do not change company names, job titles, or dates except to normalize date formatting.
   - You may rephrase descriptions for clarity and impact, combine or split bullets, and remove clearly irrelevant bullets to increase focus.
7. Tone and content:
   - Professional and factual. Avoid hyperbole and marketing-speak (e.g., avoid "world-class", "unmatched", "best-in-class").
   - Point strengths and weaknesses of the final CV only if the user explicitly requests a critique. (Default: do not append critique.)
8. Output constraints:
   - Return ONLY the formatted CV text following the Harvard template exact structure and Markdown bold section headers.
   - Do NOT include any analysis, notes, or debugging text in the output.
   - If the input cv_content is missing essential sections that prevent producing a meaningful CV (e.g., no experience and no education), produce a minimal Harvard-formatted CV using available fields and leave missing sections out.

Additional useful resources for your generation (do not print these unless requested):
- Action verb examples: led, designed, implemented, launched, reduced, increased, automated, evaluated, negotiated, coached, scaled, optimized.
- Quantifier guidance: always preserve exact numbers from cv_content; do not invent estimates.

Now: Using the Harvard CV template shown below, convert cv_content into that exact layout while applying the steps above and tailoring to:
Job Title: {job_title}
Company: {company}
{f"Job Description: {job_desc_text}" if job_desc_text else ""}

### perfect Harvard CV template (layout to copy exactly) ###
Your Name
City, State Zip • name@college.harvard.edu • phone number

**Summary**
Concise 3–4 line paragraph summarizing the candidate’s most relevant experience, core competencies, and tools aligned with the job keywords. Structure: [role/industry experience] + [core strengths] + [key technical skills/tools] + [career goal aligned with company/job]. No pronouns or subjective language.

**Experience**

ORGANIZATION City, State
Position Title Month Year – Month Year
• Beginning with your most recent position, describe your experience, skills, and resulting outcomes in bullet or paragraph form.
• Begin each line with an action verb and include details that will help the reader understand your accomplishments, skills, knowledge, abilities, or achievements.
• Quantify where possible.
• Do not use personal pronouns; each line should be a phrase rather than full sentence.

ORGANIZATION City, State
Position Title Month Year – Month Year
• [repeat format for each role]

**Education**

UNIVERSITY
Degree, Concentration. GPA [Optional] Graduation Date
Thesis [Optional]
Relevant Coursework: [Optional]

**Projects **

PROJECT NAME – [Optional brief context or employer]  
• Action verb + what you did + impact/tools (quantify where possible)  
• [Repeat bullets for up to 3–4 key projects]  

**Skills & Interests**
Technical: [List tools/languages present in cv_content and matched to job_description]
Language: [List foreign languages and proficiency levels if present]

**Certifications **

[List certifications in reverse chronological order exactly as provided. Note “In Progress” if applicable.]  

### end template ###

Convert the provided cv_content into the Harvard layout above, strictly follow formatting rules, avoid fabricating any information, and tailor language to the target job.

CV DATA TO CONVERT:
{json.dumps(cv_content, indent=2)}
"""
        
        response = llm_model.invoke([
            {"role": "system", "content": "You are a professional resume writer specializing in Harvard CV format. Create clean, professional resumes that follow the template exactly. You can also help users compare and analyze different resume versions using the read_resume_file and list_resume_files tools."},
            {"role": "user", "content": prompt}
        ])
        
        # Extract the formatted CV text
        cv_text = response.content.strip()
        
        # Create filename for PDF
        safe_company = re.sub(r'[^\w\s-]', '', company).strip()
        safe_title = re.sub(r'[^\w\s-]', '', job_title).strip()
        pdf_filename = f"Resume_{safe_company}_{safe_title}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        
        # Create data directory if it doesn't exist
        data_dir = get_data_dir()
        data_dir.mkdir(exist_ok=True)
        
        # Full path to the PDF file
        pdf_path = data_dir / pdf_filename
        
        # Generate PDF using reportlab
        from reportlab.lib.pagesizes import letter
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        
        # Create PDF document
        doc = SimpleDocTemplate(str(pdf_path), pagesize=letter)
        styles = getSampleStyleSheet()
        
        # Create custom style for the CV
        cv_style = ParagraphStyle(
            'CVStyle',
            parent=styles['Normal'],
            fontSize=10,
            leading=12,
            leftIndent=0,
            rightIndent=0,
            spaceAfter=6,
            fontName='Helvetica'
        )
        
        # Split the CV text into paragraphs and create PDF content
        story = []
        lines = cv_text.split('\n')
        
        for line in lines:
            line = line.strip()
            if line:
                # Handle different formatting based on content
                if line.startswith('**') and line.endswith('**'):  # Markdown bold section headers
                    section_title = line[2:-2]  # Remove ** markers
                    p = Paragraph(f"<b>{section_title}</b>", cv_style)
                elif line.isupper() and len(line) > 3:  # Fallback for uppercase section headers
                    p = Paragraph(f"<b>{line}</b>", cv_style)
                elif line.startswith('•'):  # Bullet points
                    p = Paragraph(f"&nbsp;&nbsp;&nbsp;&nbsp;{line}", cv_style)
                else:  # Regular text
                    p = Paragraph(line, cv_style)
                
                story.append(p)
                story.append(Spacer(1, 3))
        
        # Build PDF
        doc.build(story)
        
        return str(pdf_path)
        
    except Exception as e:
        print(f"❌ Error generating PDF: {e}")
        return ""

def extract_matching_keywords(job_title: str, keywords: Dict[str, List[str]]) -> List[str]:
    """Extract keywords that match the job title, including negative keywords."""
    title_lower = job_title.lower()
    matching = []
    
    for category, keyword_list in keywords.items():
        if isinstance(keyword_list, list):
            for keyword in keyword_list:
                if keyword.lower() in title_lower:
                    if category == "negative_keywords":
                        matching.append(f"❌{keyword}")  # Mark negative keywords with ❌
                    else:
                        matching.append(f"{category}:{keyword}")
    
    return matching

def remove_duplicate_jobs(jobs: List[dict]) -> List[dict]:
    """Remove duplicate jobs based on title, company, and location."""
    seen_jobs = set()
    unique_jobs = []
    
    for job in jobs:
        # Create a unique identifier based on title, company, and location
        # Handle cases where fields might be float, None, or other types
        title = str(job.get('title', '')).lower().strip()
        company = str(job.get('company', '')).lower().strip()
        location = str(job.get('location', '')).lower().strip()
        
        job_id = (title, company, location)
        
        if job_id not in seen_jobs:
            seen_jobs.add(job_id)
            unique_jobs.append(job)
    
    return unique_jobs

def filter_jobs_with_negative_keywords(jobs: List[dict], negative_keywords: List[str]) -> List[dict]:
    """Filter out jobs with negative keywords that should be completely excluded."""
    if not negative_keywords:
        return jobs
    
    filtered_jobs = []
    
    for job in jobs:
        title = str(job.get('title', '')).lower().strip()
        company = str(job.get('company', '')).lower().strip()
        
        # Check if job title or company contains any negative keywords
        should_exclude = False
        for keyword in negative_keywords:
            if keyword.lower() in title or keyword.lower() in company:
                should_exclude = True
                break
        
        if not should_exclude:
            filtered_jobs.append(job)
    
    return filtered_jobs

def convert_resume():
    """Convert PDF resume to structured JSON using LLM."""
    # Paths
    pdf_path = get_data_dir() / "Resume.pdf"
    output_path = get_data_dir() / "base_cv.json"
    
    # Extract text from PDF
    with pdfplumber.open(pdf_path) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    
    # Initialize OpenAI client
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    # LLM prompt
    prompt = f"""
    Parse this resume and return ONLY valid JSON. Adapt the structure based on what's actually present in the resume.
    
    Use this flexible structure:
    {{
        "summary": "Professional summary or objective (if present)",
        "skills": {{
            "group_name_1": ["skill1", "skill2", "skill3"],
            "group_name_2": ["skill1", "skill2"],
            "programming": ["Python", "JavaScript", "SQL"],
            "ai_ml": ["Machine Learning", "Deep Learning", "NLP"],
            "cloud": ["AWS", "Azure", "GCP"],
            "tools": ["Docker", "Kubernetes", "Git"],
            "languages": ["English (Native)", "Spanish (Fluent)"],
            "other": ["Any other skill category"]
        }},
        "experience": [
            {{
                "company": "Company Name",
                "title": "Job Title",
                "location": "City, Country (if available)",
                "duration": "Start Date - End Date or Present",
                "description": ["Key achievement 1", "Key achievement 2", "Key responsibility 3"]
            }}
        ],
        "education": [
            {{
                "degree": "Degree Name",
                "institution": "Institution Name", 
                "location": "City, Country (if available)",
                "year": "Graduation Year or Date Range",
                "gpa": "GPA (if mentioned)",
                "relevant_coursework": ["Course 1", "Course 2"] (optional)
            }}
        ],
        "certifications": [
            "Certification Name 1 (Year)",
            "Certification Name 2 (Year)"
        ],
        "projects": [
            {{
                "name": "Project Name",
                "description": "Brief project description",
                "technologies": ["Tech 1", "Tech 2"],
                "url": "Project URL (if available)"
            }}
        ],
        "awards": [
            "Award Name 1 (Year)",
            "Award Name 2 (Year)"
        ],
        "publications": [
            {{
                "title": "Publication Title",
                "venue": "Journal/Conference Name",
                "year": "Year",
                "authors": "Author List",
                "url": "Publication URL (if available)"
            }}
        ],
        "volunteer": [
            {{
                "role": "Volunteer Role",
                "organisation": "Organisation Name",
                "duration": "Duration",
                "description": "What you did"
            }}
        ]
    }}

    Instructions:
    1. Only include sections that exist in the resume
    2. For skills, group them logically based on the resume's organization (e.g., "technical_skills", "soft_skills", "programming_languages", "frameworks", etc.)
    3. If skills are not grouped in the resume, create logical groups
    4. For experience, include all work experience, internships, and relevant positions
    5. Extract key achievements and responsibilities, not just job descriptions
    6. If dates are unclear, use your best judgment
    7. Include any additional sections that might be relevant (projects, publications, awards, etc.)
    8. If a section is not present, omit it from the JSON
    9. Ensure all dates, names, and details are accurate to the source

    Resume text:
    {text}
    """
    
    # Call LLM
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are an expert at parsing resumes. Return only valid JSON."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.1
    )
    
    # Extract and clean JSON
    content = response.choices[0].message.content
    data = parse_json_response(content, "resume conversion")
    
    # Add metadata
    data['raw_text'] = text
    data['conversion_date'] = datetime.now().isoformat()
    data['source_file'] = str(pdf_path)
    
    # Save to file
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"Resume converted and saved to: {output_path}")

# ============================================================================
# DATA INITIALIZATION
# ============================================================================
def build_resume_text(resume):
    sections = []

    # Add summary
    if "summary" in resume:
        sections.append(resume["summary"])

    # Add skills
    if "skills" in resume:
        for category, items in resume["skills"].items():
            sections.append(" ".join(items))

    # Add experience
    if "experience" in resume:
        for exp in resume["experience"]:
            exp_text = f"{exp['title']} at {exp['company']} - " + " ".join(exp["description"])
            sections.append(exp_text)

    # Add education
    if "education" in resume:
        for edu in resume["education"]:
            sections.append(f"{edu['degree']} at {edu['institution']}")

    return "\n".join(sections)

# Load resume data function
def load_resume_data():
    """Load resume data from existing base_cv.json file."""
    # Load resume data
    resume = json.load(open("data/base_cv.json"))
    resume_text = build_resume_text(resume)
    return resume, resume_text

# Initialize resume data (will be loaded in chatbot_mode)
resume = None
resume_text = None

# Load preferences and concatenate into single string
preferences = json.load(open("data/preferences.json"))
pref_text = " ".join(preferences[0].get("roles", []) +
                     preferences[0].get("location", []) +
                     preferences[0].get("companies", []))

# Initialize embeddings (will be created when needed)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
resume_vec = None
pref_vec = None

# ============================================================================
# AGENT INITIALIZATION
# ============================================================================

# Create the agent with our tools (including new agentic capabilities)
agent = create_react_agent(
    model=ChatOpenAI(model="gpt-4o-mini"),
    tools=[
        # Core job search tools
        jobspy_search, find_best_job_matches, fetch_job_description, analyze_job_requirements,
        # CV customization tools
        customize_cv, generate_cv_json, generate_cv_pdf, read_resume_file, list_resume_files, 
        explain_cv_modifications, create_customized_resume,
        # Strategic agentic capabilities (simplified)
        browse_web, reflect_on_information
    ]
)

# Chat memory for ongoing conversations
chat_memory = []

def cleanup_chat_memory():
    """Clean up chat memory to prevent unlimited growth while preserving important context."""
    global chat_memory
    
    # Keep memory under 100 messages to prevent token overflow
    if len(chat_memory) > 100:
        # Keep first 10 messages (initial context) and last 50 messages (recent context)
        important_messages = chat_memory[:10]  # Initial context
        recent_messages = chat_memory[-50:]    # Recent context
        
        # Remove duplicates and combine
        seen = set()
        cleaned_memory = []
        
        for msg in important_messages + recent_messages:
            msg_key = f"{msg['role']}:{msg['content'][:100]}"
            if msg_key not in seen:
                seen.add(msg_key)
                cleaned_memory.append(msg)
        
        chat_memory = cleaned_memory
        print(f"🧹 Memory cleaned: {len(chat_memory)} messages retained")

def get_initial_job_search_prompt():
    """Get the initial job search prompt."""
    # Default flexible prompt
    default_prompt = """Find AI related job openings in London, UK that were published in the last 24 hours. Only include jobs listed on LinkedIn, Indeed and Google. 

Then analyze these jobs and find the best matches based on my resume and preferences using similarity matching with keyword reranking. 

    Please display the results in a clean, readable format (NOT a table). For each job, show:
    - Job Title at Company Name
    - Location
    - Date Posted: [publishing date]
    - Overall Score: X.XXX (Similarity: X.XXX, Keywords: X.XXX)
    - Resume Match: X.XXX, Preferences Match: X.XXX
    - Matching Keywords: [list of matched keywords]
    - Job URL: [link]

    Show me the top 20 most relevant job opportunities with their detailed analysis in this format."""
    
    print("\n" + "="*60)
    print("INITIAL JOB SEARCH PROMPT:")
    print("="*60)
    print(default_prompt)
    print("="*60)
    
    modify = input("\nWould you like to modify this prompt? (y/n): ").strip().lower()
    
    if modify in ['y', 'yes']:
        print("\nEnter your custom prompt (press Enter twice when done):")
        lines = []
        while True:
            line = input()
            if line == "" and lines and lines[-1] == "":
                break
            lines.append(line)
        
        custom_prompt = "\n".join(lines[:-1])  # Remove the last empty line
        if custom_prompt.strip():
            return custom_prompt.strip()
    
    return default_prompt

def run_agent_with_memory(user_message):
    """Run the agent with enhanced memory and agentic capabilities."""
    # Clean up memory if needed
    cleanup_chat_memory()
    
    # Add user message to memory
    chat_memory.append({"role": "user", "content": user_message, "timestamp": datetime.now().isoformat()})
    
    # Prepare messages for the agent
    messages = []
    
    # Enhanced system context with agentic capabilities
    system_context = """You are an advanced AI job search agent with autonomous decision-making capabilities.

CORE IDENTITY:
You are a strategic, intelligent agent that can browse the internet, reflect on information, and make informed decisions to help users with job searching and career development.

KEY CAPABILITIES:
- Search for jobs across multiple platforms (LinkedIn, Indeed, Google)
- Find best job matches using AI similarity matching and keyword analysis
- Create customized CVs for specific job applications
- Generate professional PDF resumes using Harvard template
- Browse the internet for real-time information and market insights
- Reflect on gathered information to make strategic decisions
- Plan and execute multi-step workflows autonomously

STRATEGIC TOOLS (USE SPARINGLY):
1. **browse_web**: Search the internet for current information
   - ONLY use when user specifically asks for current market data, salary information, or company research
   - Use for: "What's the current salary for X role?", "Tell me about company Y", "What are the latest trends in Z industry?"
   - DO NOT use for general job search or CV creation unless specifically requested
2. **reflect_on_information**: Analyze gathered information
   - ONLY use when you have gathered information that needs strategic analysis
   - Use to explain complex decisions or provide deeper insights
   - DO NOT use for simple, straightforward requests

IMPORTANT: Use these tools ONLY when they add clear value. For most requests, use the standard job search and CV tools. Avoid unnecessary tool chaining.

ENHANCED WORKFLOWS (ONLY WHEN REQUESTED):
- For salary questions: Use browse_web to get current market data
- For company research: Use browse_web to gather company information
- For industry trends: Use browse_web to find current market insights
- For complex analysis: Use reflect_on_information to provide deeper insights

MEMORY INTEGRATION:
- Job results are stored in memory with detailed information
- CV customizations are tracked with modification explanations
- Use this context to answer follow-up questions intelligently
- Build on previous conversations and decisions

CRITICAL INSTRUCTIONS:
- Answer questions directly when possible using existing information
- Only use browse_web or reflect_on_information when they add clear value
- Avoid calling multiple tools in sequence unless absolutely necessary
- If you have enough information to answer, provide the answer directly
- Do not chain tools unnecessarily - this can cause recursion errors

Be strategic and helpful, but efficient. Use the right tool for the right job."""
    
    messages.append({"role": "system", "content": system_context})
    
    # Include recent conversation history with smart memory management
    recent_messages = chat_memory[-20:]  # Get last 20 messages
    
    # Smart memory: prioritize important messages and truncate if needed
    for msg in recent_messages:
        if msg["role"] in ["user", "assistant"]:
            content = msg["content"]
            # Truncate very long messages to prevent token overflow
            if len(content) > 2000:
                content = content[:2000] + "... [truncated]"
            messages.append({"role": msg["role"], "content": content})
    
    # Show progress indicator for non-job search requests
    progress = None
    is_follow_up = any("job" in msg.get("content", "").lower() for msg in chat_memory[-3:]) and len(chat_memory) > 5
    
    if "job" not in user_message.lower() or "search" not in user_message.lower():
        if not is_follow_up:
            progress = ProgressIndicator("Processing your request")
            progress.start()
    
    try:
        result = agent.invoke({"messages": messages}, config={"recursion_limit": 25})
        
        # Extract and store the agent's response
        if 'messages' in result:
            for message in reversed(result['messages']):
                if hasattr(message, 'content') and message.content and len(message.content) > 50:
                    # Store assistant response in memory
                    chat_memory.append({"role": "assistant", "content": message.content, "timestamp": datetime.now().isoformat()})
                    if progress:
                        progress.stop()
                    return message.content
        
        if progress:
            progress.stop()
        return "I apologize, but I couldn't process your request properly."
    
    except Exception as e:
        if progress:
            progress.stop()
        print(f"\n❌ Error: {e}")
        return "I encountered an error while processing your request."
    
    finally:
        if progress:
            progress.stop()

def show_initial_info():
    """Display initial information about the agent and requirements."""
    print("🤖 AI Job Search Chatbot v5 - Enhanced with CV Customization")
    print("="*60)
    print("📋 FEATURES:")
    print("• Searches 200+ job openings across multiple job sites")
    print("• AI-powered job matching with your resume and preferences")
    print("• Customized CV generation for specific job applications")
    print("• Professional PDF resume creation using Harvard template")
    print("="*60)

def store_cv_in_memory(cv_data: Dict[str, Any], job_title: str, company: str, file_path: str):
    """Store CV data in chat memory for dynamic querying."""
    cv_summary = {
        "type": "customized_cv",
        "job_title": job_title,
        "company": company,
        "file_path": file_path,
        "timestamp": datetime.now().isoformat(),
        "cv_data": cv_data
    }
    
    chat_memory.append({
        "role": "system",
        "content": f"Customized CV created for {job_title} at {company}. File saved to: {file_path}"
    })
    
    chat_memory.append({
        "role": "system",
        "content": f"CV_DATA: {json.dumps(cv_summary, indent=2)}"
    })

def chatbot_mode():
    """Run in streamlined chatbot mode with dynamic LLM-driven responses."""
    show_initial_info()
    
    # Check if resume PDF exists
    resume_pdf_path = get_data_dir() / "Resume.pdf"
    if not resume_pdf_path.exists():
        print("\n❌ ERROR: Resume.pdf not found!")
        print("Please ensure Resume.pdf is in the data folder.")
        return
    
    print(f"\n✅ Resume PDF found: {resume_pdf_path.name}")
    
    # Extract fresh resume from PDF
    progress = ProgressIndicator("Extracting resume from PDF")
    progress.start()
    convert_resume()
    progress.stop()
    print("✅ Profile ready!")
    
    # Load resume data and create embeddings
    global resume, resume_text, resume_vec, pref_vec
    resume, resume_text = load_resume_data()
    resume_vec = embeddings.embed_query(resume_text)
    pref_vec = embeddings.embed_query(pref_text)
    
    # Store resume info in memory
    chat_memory.append({
        "role": "system",
        "content": f"Resume loaded: {resume.get('summary', 'N/A')[:100]}... Skills: {', '.join(list(resume.get('skills', {}).keys())[:5])}"
    })
    
    # Start with job search
    print("\n🎯 Let's start with the job search!")
    initial_prompt = get_initial_job_search_prompt()
    
    print(f"\n🔍 Running initial job search...")
    print("-" * 40)
    
    response = run_agent_with_memory(initial_prompt)
    
    print("\n📋 JOB SEARCH RESULTS:")
    print("-" * 50)
    print(response)
    print("\n" + "="*60)
    print("✅ Initial job search completed!")
    print("="*60)
    
    # Store job results in memory
    try:
        job_results_path = get_data_dir() / "job_results_v5.json"
        if job_results_path.exists():
            with open(job_results_path, 'r', encoding='utf-8') as f:
                job_data = json.load(f)
                job_results = job_data.get('jobs', [])
                if job_results:
                    store_job_results_in_memory(job_results)
    except Exception as e:
        print(f"⚠️ Could not load job results: {e}")
    
    # Chat loop
    print("\n💬 You can now ask me anything about the jobs, request CV customization, or get career advice!")
    print("💡 Try: 'Show me the top 5 jobs', 'Create a CV for job #3', 'Compare my CVs', 'Explain the modifications'")
    
    while True:
        print("\n" + "="*50)
        user_input = input("You: ").strip()
        
        if user_input.lower() in ['exit', 'quit', 'bye']:
            print("👋 Goodbye! Thanks for using the AI Job Search Chatbot!")
            break
        elif user_input.lower() == 'clear':
            chat_memory.clear()
            print("🧹 Chat memory cleared!")
            continue
        elif user_input.lower() == 'help':
            print("\n📚 I can help you with:")
            print("• Job search and analysis: 'Find AI jobs in London', 'Show me the top jobs'")
            print("• CV customization: 'Create a CV for job #3', 'Customize my CV for [company]'")
            print("• CV analysis: 'Compare my CVs', 'Explain the modifications', 'List my resumes'")
            print("• Career advice: 'What skills should I highlight?', 'How can I improve my CV?'")
            print("• General queries: Ask me anything about the jobs or your career!")
            continue
        elif not user_input:
            continue
        
        # Process user input
        response = run_agent_with_memory(user_input)
        
        print(f"\n🤖 Assistant: {response}")
        
        # Check if CV was created and store in memory
        if any(keyword in response.lower() for keyword in ["customized cv", "cv generated", "pdf generated", "resume created"]):
            try:
                import re
                file_paths = re.findall(r'[A-Za-z]:\\[^\\]+\\.pdf|[A-Za-z]:\\[^\\]+\\.json', response)
                if file_paths:
                    chat_memory.append({
                        "role": "system",
                        "content": f"Recent CV created: {file_paths[0]}"
                    })
            except:
                pass

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    chatbot_mode()
