"""
FastAPI Main Application
ATS Resume Analyzer API
"""
import os
import json
import datetime
from datetime import datetime, timedelta
from typing import Optional, List
from pathlib import Path

from fastapi import FastAPI, Depends, HTTPException, status, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from pydantic import BaseModel, EmailStr
from pydantic.config import ConfigDict
from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi.staticfiles import StaticFiles

from database import (
    get_db, 
    init_db_sync, 
    User, 
    Resume, 
    JobDescription, 
    Analysis,
    settings
)

# ============== Initialize App ==============
app = FastAPI(
    title="ATS Resume Analyzer API",
    description="AI-powered ATS resume analysis and optimization",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    origins = [
    "https://ats-frontend-zeta.vercel.app",
    "http://localhost:5173",
]

    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="api/auth/login")

# ============== Pydantic Models ==============

class UserCreate(BaseModel):
    email: EmailStr
    username: str
    password: str
    full_name: Optional[str] = None


class UserResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    email: str
    username: str
    full_name: Optional[str]
    is_active: bool
    is_verified: bool
    created_at: datetime


class Token(BaseModel):
    access_token: str
    token_type: str
    user: UserResponse


class ResumeCreate(BaseModel):
    filename: str
    file_content: str
    skills: Optional[str] = None
    experience_years: Optional[int] = None


class ResumeResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    filename: str
    file_content: Optional[str]
    skills: Optional[str]
    experience_years: Optional[int]
    created_at: datetime


class JobDescriptionCreate(BaseModel):
    title: str
    company: Optional[str] = None
    description: str
    requirements: Optional[str] = None


class JobDescriptionResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    title: str
    company: Optional[str]
    description: str
    requirements: Optional[str]
    created_at: datetime


class AnalysisCreate(BaseModel):
    resume_id: int
    job_description_id: int
    ats_score: float
    keyword_coverage: Optional[float] = None
    skill_overlap: Optional[float] = None
    semantic_alignment: Optional[float] = None
    formatting_score: Optional[float] = None
    gaps: Optional[str] = None
    matches: Optional[str] = None
    recommendations: Optional[str] = None


class AnalysisResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    ats_score: float
    keyword_coverage: Optional[float]
    skill_overlap: Optional[float]
    semantic_alignment: Optional[float]
    formatting_score: Optional[float]
    gaps: Optional[str]
    matches: Optional[str]
    recommendations: Optional[str]
    created_at: datetime
    resume: Optional[ResumeResponse] = None
    job_description: Optional[JobDescriptionResponse] = None


class AnalysisRequest(BaseModel):
    resume_text: str
    job_description: str

class AnalysisHistoryResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    ats_score: float
    keyword_coverage: Optional[float]
    skill_overlap: Optional[float]
    semantic_alignment: Optional[float]
    formatting_score: Optional[float]
    gaps: Optional[str]
    matches: Optional[str]
    recommendations: Optional[str]
    created_at: datetime
    # Don't include resume and job_description as nested objects
    resume_id: Optional[int] = None
    job_description_id: Optional[int] = None    


# ============== Helper Functions ==============

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash."""
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str):
    password = password[:72]
    return pwd_context.hash(password)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Create JWT access token."""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)
    return encoded_jwt


async def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: AsyncSession = Depends(get_db)
) -> User:
    """Get current authenticated user."""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    
    result = await db.execute(select(User).where(User.email == email))
    user = result.scalar_one_or_none()
    
    if user is None:
        raise credentials_exception
    return user


# ============== Authentication Endpoints ==============

@app.post("/api/auth/register", response_model=UserResponse)
async def register(user_data: UserCreate, db: AsyncSession = Depends(get_db)):
    """Register a new user."""
    # Check if user exists
    result = await db.execute(
        select(User).where((User.email == user_data.email) | (User.username == user_data.username))
    )
    existing_user = result.scalar_one_or_none()
    
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email or username already registered"
        )
    
    # Create new user
    hashed_password = get_password_hash(user_data.password)
    new_user = User(
        email=user_data.email,
        username=user_data.username,
        hashed_password=hashed_password,
        full_name=user_data.full_name
    )
    
    db.add(new_user)
    await db.commit()
    await db.refresh(new_user)
    
    return new_user


@app.post("/api/auth/login", response_model=Token)
async def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: AsyncSession = Depends(get_db)
):
    """Login user and return access token."""
    result = await db.execute(select(User).where(User.email == form_data.username))
    user = result.scalar_one_or_none()
    
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is disabled"
        )
    
    access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.email}, expires_delta=access_token_expires
    )
    
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": UserResponse.model_validate(user)
    }


@app.get("/api/auth/me", response_model=UserResponse)
async def get_current_user_info(current_user: User = Depends(get_current_user)):
    """Get current user information."""
    return current_user


# ============== Resume Endpoints ==============

@app.post("/api/resumes", response_model=ResumeResponse)
async def create_resume(
    resume_data: ResumeCreate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Create a new resume record."""
    # Ensure upload directory exists
    upload_dir = Path("data/uploads")
    upload_dir.mkdir(parents=True, exist_ok=True)
    
    # Save file
    file_path = upload_dir / f"{current_user.id}_{resume_data.filename}"
    
    new_resume = Resume(
        user_id=current_user.id,
        filename=resume_data.filename,
        file_path=str(file_path),
        file_content=resume_data.file_content,
        skills=resume_data.skills,
        experience_years=resume_data.experience_years
    )
    
    db.add(new_resume)
    await db.commit()
    await db.refresh(new_resume)
    
    return new_resume


@app.get("/api/resumes", response_model=List[ResumeResponse])
async def get_resumes(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get all resumes for current user."""
    result = await db.execute(
        select(Resume).where(Resume.user_id == current_user.id).order_by(Resume.created_at.desc())
    )
    resumes = result.scalars().all()
    return resumes


@app.get("/api/resumes/{resume_id}", response_model=ResumeResponse)
async def get_resume(
    resume_id: int,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get a specific resume."""
    result = await db.execute(
        select(Resume).where(
            Resume.id == resume_id,
            Resume.user_id == current_user.id
        )
    )
    resume = result.scalar_one_or_none()
    
    if not resume:
        raise HTTPException(status_code=404, detail="Resume not found")
    
    return resume


@app.delete("/api/resumes/{resume_id}")
async def delete_resume(
    resume_id: int,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Delete a resume."""
    result = await db.execute(
        select(Resume).where(
            Resume.id == resume_id,
            Resume.user_id == current_user.id
        )
    )
    resume = result.scalar_one_or_none()
    
    if not resume:
        raise HTTPException(status_code=404, detail="Resume not found")
    
    await db.delete(resume)
    await db.commit()
    
    return {"message": "Resume deleted successfully"}


# ============== Job Description Endpoints ==============

@app.post("/api/job-descriptions", response_model=JobDescriptionResponse)
async def create_job_description(
    jd_data: JobDescriptionCreate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Create a new job description record."""
    new_jd = JobDescription(
        title=jd_data.title,
        company=jd_data.company,
        description=jd_data.description,
        requirements=jd_data.requirements
    )
    
    db.add(new_jd)
    await db.commit()
    await db.refresh(new_jd)
    
    return new_jd


@app.get("/api/job-descriptions", response_model=List[JobDescriptionResponse])
async def get_job_descriptions(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get all job descriptions."""
    result = await db.execute(select(JobDescription).order_by(JobDescription.created_at.desc()))
    jds = result.scalars().all()
    return jds


# ============== Analysis Endpoints ==============

@app.post("/api/analyses", response_model=AnalysisResponse)
async def create_analysis(
    analysis_data: AnalysisCreate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Save an analysis result."""
    new_analysis = Analysis(
        user_id=current_user.id,
        resume_id=analysis_data.resume_id,
        job_description_id=analysis_data.job_description_id,
        ats_score=analysis_data.ats_score,
        keyword_coverage=analysis_data.keyword_coverage,
        skill_overlap=analysis_data.skill_overlap,
        semantic_alignment=analysis_data.semantic_alignment,
        formatting_score=analysis_data.formatting_score,
        gaps=analysis_data.gaps,
        matches=analysis_data.matches,
        recommendations=analysis_data.recommendations
    )
    
    db.add(new_analysis)
    await db.commit()
    await db.refresh(new_analysis)
    
    # Fetch related data
    result = await db.execute(select(Analysis).where(Analysis.id == new_analysis.id))
    analysis = result.scalar_one()
    
    return analysis


@app.get("/api/analyses", response_model=List[AnalysisResponse])
async def get_analyses(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get all analyses for current user."""
    from sqlalchemy.orm import selectinload
    
    # Eagerly load relationships to avoid greenlet error
    result = await db.execute(
        select(Analysis)
        .options(
            selectinload(Analysis.resume),
            selectinload(Analysis.job_description)
        )
        .where(Analysis.user_id == current_user.id)
        .order_by(Analysis.created_at.desc())
    )
    analyses = result.scalars().all()
    
    # Convert to response model
    response_data = []
    for analysis in analyses:
        # Manually construct response to ensure relationships are loaded
        analysis_dict = {
            "id": analysis.id,
            "ats_score": analysis.ats_score,
            "keyword_coverage": analysis.keyword_coverage,
            "skill_overlap": analysis.skill_overlap,
            "semantic_alignment": analysis.semantic_alignment,
            "formatting_score": analysis.formatting_score,
            "gaps": analysis.gaps,
            "matches": analysis.matches,
            "recommendations": analysis.recommendations,
            "created_at": analysis.created_at,
            "resume": analysis.resume,
            "job_description": analysis.job_description
        }
        response_data.append(analysis_dict)
    
    return response_data

@app.get("/api/analyses/{analysis_id}", response_model=AnalysisResponse)
async def get_analysis(
    analysis_id: int,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get a specific analysis."""
    result = await db.execute(
        select(Analysis).where(
            Analysis.id == analysis_id,
            Analysis.user_id == current_user.id
        )
    )
    analysis = result.scalar_one_or_none()
    
    if not analysis:
        raise HTTPException(status_code=404, detail="Analysis not found")
    
    return analysis


@app.delete("/api/analyses/{analysis_id}", status_code=204)
async def delete_analysis(
    analysis_id: int,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Delete a specific analysis."""
    result = await db.execute(
        select(Analysis).where(
            Analysis.id == analysis_id,
            Analysis.user_id == current_user.id
        )
    )
    analysis = result.scalar_one_or_none()
    
    if not analysis:
        raise HTTPException(status_code=404, detail="Analysis not found")
    
    await db.delete(analysis)
    await db.commit()
    
    return None


# ============== ATS Analysis Endpoint ==============

# Replace the analyze_resume function in main.py with this improved version

@app.post("/api/analyze")
async def analyze_resume(
    resume_text: str = Form(...),
    job_description: str = Form(...),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Analyze a resume against a job description using the ATS pipeline.
    """
    try:
        # Import pipeline modules
        import sys
        import os
        project_root = os.path.dirname(os.path.abspath(__file__))
        sys.path.insert(0, project_root)
        
        from pipeline.utils import read_config
        from pipeline.ingest import load_resume, load_jd
        from pipeline.normalize import normalize_resume, normalize_jd
        from pipeline.embeddings import pairwise_match, build_embeddings
        from pipeline.skills_taxonomy import load_taxonomy, extract_and_normalize_skills
        from pipeline.keyphrases import extract_keyphrases_hybrid
        from pipeline.scoring import compute_features, combined_score
        from pipeline.gap_analyzer import rank_missing_by_impact
        
        # Load config
        cfg = read_config("config.yaml")
        
        # Create temporary files for analysis
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as resume_file:
            resume_file.write(resume_text)
            resume_path = resume_file.name
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as jd_file:
            jd_file.write(job_description)
            jd_path = jd_file.name
        
        try:
            # Run analysis
            resume = load_resume(resume_path)
            jd = load_jd(jd_path)
            resume_n = normalize_resume(resume)
            jd_n = normalize_jd(jd)
            
            embs = build_embeddings(cfg)
            matches = pairwise_match(jd_n["requirements"], resume_n["bullets"], cfg, embs)
            taxonomy = load_taxonomy(cfg["paths"]["skills_taxonomy"])
            jd_skills, res_skills = extract_and_normalize_skills(jd_n, resume_n, taxonomy)
            jd_kp = extract_keyphrases_hybrid(jd_n["text"], jd_n["requirements"])
            res_kp = extract_keyphrases_hybrid(resume_n["full_text"], resume_n["bullets"])
            
            feats = compute_features(jd_n, resume_n, matches, jd_skills, res_skills, jd_kp, res_kp, embs)
            score = combined_score(feats, cfg)
            gaps = rank_missing_by_impact(feats, jd_skills, res_skills, jd_kp, res_kp, cfg)
            
            # Format gaps properly
            formatted_gaps = []
            for gap in gaps[:15]:  # Top 15 gaps
                formatted_gaps.append({
                    "skill": gap.get("term", ""),
                    "impact": f"+{gap.get('estimated_gain', 1.0)}%"
                })
            
            # Format matches properly
            formatted_matches = []
            for match in matches[:10]:  # Top 10 matches
                formatted_matches.append({
                    "requirement": match.get("jd_req", ""),
                    "matched_bullet": match.get("bullet", ""),
                    "score": round(match.get("cross", match.get("cos", 0)) * 100, 2)
                })
            
            # Generate recommendations
            recommendations = []
            for gap in formatted_gaps[:8]:
                recommendations.append({
                    "skill": gap["skill"],
                    "suggestion": f"Add '{gap['skill']}' to your resume to improve your match"
                })
            
            # Save analysis to database
            # First, create or get job description
            jd_record = JobDescription(
                title="Job Description Analysis",
                description=job_description[:5000],  # Limit length
                requirements=json.dumps(jd_n.get("requirements", [])[:20])
            )
            db.add(jd_record)
            await db.flush()
            
            # Create resume record
            resume_record = Resume(
                user_id=current_user.id,
                filename="analysis_resume.txt",
                file_path=resume_path,
                file_content=resume_text[:5000],  # Limit length
                skills=json.dumps(list(res_skills)[:50])
            )
            db.add(resume_record)
            await db.flush()
            
            # Create analysis record
            analysis_record = Analysis(
                user_id=current_user.id,
                resume_id=resume_record.id,
                job_description_id=jd_record.id,
                ats_score=score,
                keyword_coverage=feats["features"].get("keyword_coverage", 0),
                skill_overlap=feats["features"].get("skill_overlap", 0),
                semantic_alignment=feats["features"].get("semantic_alignment", 0),
                formatting_score=feats["features"].get("formatting", 0),
                gaps=json.dumps(formatted_gaps),
                matches=json.dumps(formatted_matches),
                recommendations=json.dumps(recommendations)
            )
            db.add(analysis_record)
            await db.commit()
            
            # Prepare response
            result = {
                "success": True,
                "ats_score": round(score, 2),
                "subscores": {
                    "keyword_coverage": round(feats["features"].get("keyword_coverage", 0), 2),
                    "skill_overlap": round(feats["features"].get("skill_overlap", 0), 2),
                    "semantic_alignment": round(feats["features"].get("semantic_alignment", 0), 2),
                    "formatting": round(feats["features"].get("formatting", 0), 2)
                },
                "gaps": formatted_gaps,
                "matches": formatted_matches,
                "recommendations": recommendations,
                "resume_text": resume_n.get("full_text", "")[:1000],
                "jd_requirements": jd_n.get("requirements", [])[:15],
                "analysis_id": analysis_record.id
            }
            
            return result
            
        finally:
            # Cleanup temp files
            try:
                os.unlink(resume_path)
                os.unlink(jd_path)
            except:
                pass
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {str(e)}"
        )

@app.get("/api/analyses", response_model=List[AnalysisHistoryResponse])
async def get_analyses(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    result = await db.execute(
        select(Analysis)
        .where(Analysis.user_id == current_user.id)
        .order_by(Analysis.created_at.desc())
    )
    analyses = result.scalars().all()
    return analyses

#============ Optimize Resume Endpoint ==============
@app.post("/api/optimize")
async def optimize_resume(
    resume_text: str = Form(...),
    job_description: str = Form(...),
    missing_keywords: str = Form(...),  # JSON string of missing keywords
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Optimize a resume using AI to include missing keywords and improve formatting.
    """
    try:
        # Import pipeline modules
        import sys
        import os
        import json
        project_root = os.path.dirname(os.path.abspath(__file__))
        sys.path.insert(0, project_root)
        
        from pipeline.utils import read_config
        from pipeline.rewrite_llm import optimize_resume_with_llm
        
        # Load config
        cfg = read_config("config.yaml")
        
        # Parse missing keywords
        try:
            keywords = json.loads(missing_keywords)
            if isinstance(keywords, list):
                # Extract just the skill names
                missing_skills = [k.get("skill", k) if isinstance(k, dict) else k for k in keywords[:10]]
            else:
                missing_skills = []
        except:
            missing_skills = []
        
        # Call the optimization function
        optimized_text = optimize_resume_with_llm(
            resume_text=resume_text,
            job_description=job_description,
            missing_keywords=missing_skills,
            cfg=cfg
        )
        
        return {
            "success": True,
            "optimized_text": optimized_text,
            "keywords_used": missing_skills
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Optimization failed: {str(e)}"
        )

# ============== Health Check ==============

@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "ATS Resume Analyzer API"}


# ============== Startup Event (Using modern lifespan) ==============

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize database on startup."""
    init_db_sync()
    print("Database initialized successfully!")
    yield

app.router.lifespan_context = lifespan


# ============== Run Instructions ==============

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)




