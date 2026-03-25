import { useUserRole } from "./UserRoleContext";
import "./cssFile/RoleSelector.css";

function RoleSelector() {
  const { setUserRole } = useUserRole();

  return (
    <div className="role-selector-page">
      <div className="role-selector-overlay"></div>
      <div className="role-selector-content">
        <div className="role-selector-brand">
          <span className="role-brand-icon">{"\u25C6"}</span>
          <span className="role-brand-name">SkillSync</span>
        </div>
        <h1 className="role-selector-title">Welcome to SkillSync</h1>
        <p className="role-selector-subtitle">
          AI-Powered Recruiting Platform &mdash; Choose how you'd like to use SkillSync
        </p>

        <div className="role-cards">
          <button
            className="role-card role-card-candidate"
            onClick={() => setUserRole("candidate")}
          >
            <div className="role-card-icon">
              <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2" />
                <circle cx="12" cy="7" r="4" />
              </svg>
            </div>
            <h2 className="role-card-title">I'm a Candidate</h2>
            <p className="role-card-desc">
              Analyze your resume, get AI coaching, interview prep, and personalized learning paths
            </p>
            <ul className="role-card-features">
              <li>Resume Analysis & Scoring</li>
              <li>AI Interview Prep</li>
              <li>AI Resume Coach</li>
              <li>Personalized Learning Path</li>
            </ul>
            <span className="role-card-cta">Continue as Candidate</span>
          </button>

          <button
            className="role-card role-card-recruiter"
            onClick={() => setUserRole("recruiter")}
          >
            <div className="role-card-icon">
              <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                <path d="M16 21v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2" />
                <circle cx="9" cy="7" r="4" />
                <path d="M22 21v-2a4 4 0 0 0-3-3.87" />
                <path d="M16 3.13a4 4 0 0 1 0 7.75" />
              </svg>
            </div>
            <h2 className="role-card-title">I'm a Recruiter</h2>
            <p className="role-card-desc">
              Evaluate candidates, batch analyze resumes, rank talent, and get hiring insights
            </p>
            <ul className="role-card-features">
              <li>Batch Resume Analysis</li>
              <li>Candidate Rankings & Comparison</li>
              <li>Credibility & Culture Assessment</li>
              <li>Role-Fit Narrative & Hiring Recommendations</li>
            </ul>
            <span className="role-card-cta">Continue as Recruiter</span>
          </button>
        </div>
      </div>
    </div>
  );
}

export default RoleSelector;
