import { useUserRole } from "./UserRoleContext";
import "./cssFile/TabNav.css";

const ALL_TABS = [
  { id: "analyze", label: "Analyze", icon: "\uD83D\uDD0D", roles: ["candidate", "recruiter"] },
  { id: "batch", label: "Batch", icon: "\uD83D\uDCC2", roles: ["recruiter"] },
  { id: "candidates", label: "Candidates", icon: "\uD83D\uDC65", roles: ["recruiter"] },
  { id: "rankings", label: "Rankings", icon: "\uD83C\uDFC6", roles: ["recruiter"] },
  { id: "compare", label: "Compare", icon: "\u2696\uFE0F", roles: ["recruiter"] },
  { id: "jd-parser", label: "JD Parser", icon: "\uD83D\uDCCB", roles: ["candidate", "recruiter"] },
  { id: "dashboard", label: "Dashboard", icon: "\uD83D\uDCCA", roles: ["candidate", "recruiter"] },
];

export function getVisibleTabs(role) {
  return ALL_TABS.filter((tab) => tab.roles.includes(role));
}

function TabNav({ activeTab, onTabChange }) {
  const { userRole, logout } = useUserRole();
  const visibleTabs = getVisibleTabs(userRole);

  return (
    <nav className="navbar">
      <div className="navbar-inner">
        <div className="navbar-links">
          {visibleTabs.map((tab) => (
            <a
              key={tab.id}
              href={"#" + tab.id}
              className={`navbar-link ${activeTab === tab.id ? "active" : ""}`}
              onClick={(e) => {
                e.preventDefault();
                onTabChange(tab.id);
              }}
            >
              <span className="navbar-link-icon">{tab.icon}</span>
              {tab.label}
            </a>
          ))}
        </div>

        <div className="navbar-right">
          <span className={`role-badge role-badge-${userRole}`}>
            {userRole === "candidate" ? "Candidate" : "Recruiter"}
          </span>
          <button className="switch-role-btn" onClick={logout} title="Switch Role">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4" />
              <polyline points="16 17 21 12 16 7" />
              <line x1="21" y1="12" x2="9" y2="12" />
            </svg>
            Switch Role
          </button>
        </div>
      </div>
    </nav>
  );
}

export default TabNav;
