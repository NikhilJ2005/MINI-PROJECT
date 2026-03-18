import "./cssFile/TabNav.css";

const tabs = [
  { id: "analyze", label: "Analyze", icon: "\uD83D\uDD0D" },
  { id: "batch", label: "Batch", icon: "\uD83D\uDCC2" },
  { id: "candidates", label: "Candidates", icon: "\uD83D\uDC65" },
  { id: "rankings", label: "Rankings", icon: "\uD83C\uDFC6" },
  { id: "compare", label: "Compare", icon: "\u2696\uFE0F" },
  { id: "jd-parser", label: "JD Parser", icon: "\uD83D\uDCCB" },
  { id: "dashboard", label: "Dashboard", icon: "\uD83D\uDCCA" },
];

function TabNav({ activeTab, onTabChange }) {
  return (
    <nav className="tab-nav">
      <div className="tab-nav-inner">
        {tabs.map((tab) => (
          <button
            key={tab.id}
            className={`tab-btn ${activeTab === tab.id ? "active" : ""}`}
            onClick={() => onTabChange(tab.id)}
          >
            <span className="tab-icon">{tab.icon}</span>
            {tab.label}
          </button>
        ))}
      </div>
    </nav>
  );
}

export default TabNav;
