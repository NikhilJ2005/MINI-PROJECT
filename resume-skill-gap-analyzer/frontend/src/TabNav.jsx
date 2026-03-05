import "./cssFile/TabNav.css";

const tabs = [
  { id: "analyze", label: "Analyze" },
  { id: "batch", label: "Batch Upload" },
  { id: "candidates", label: "Candidates" },
  { id: "rankings", label: "Rankings" },
  { id: "compare", label: "Compare" },
  { id: "jd-parser", label: "JD Parser" },
  { id: "dashboard", label: "Dashboard" },
];

function TabNav({ activeTab, onTabChange }) {
  return (
    <nav className="tab-nav">
      {tabs.map((tab) => (
        <button
          key={tab.id}
          className={`tab-btn ${activeTab === tab.id ? "active" : ""}`}
          onClick={() => onTabChange(tab.id)}
        >
          {tab.label}
        </button>
      ))}
    </nav>
  );
}

export default TabNav;
