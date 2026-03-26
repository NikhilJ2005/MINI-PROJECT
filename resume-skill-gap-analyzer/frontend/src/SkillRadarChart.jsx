import { RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar, ResponsiveContainer, Legend } from "recharts";
import "./cssFile/SkillRadarChart.css";

function SkillRadarChart({ report }) {
  if (!report || !report.skill_breakdown) return null;

  const required = report.skill_breakdown.required_analysis || [];
  const niceToHave = report.skill_breakdown.nice_to_have_analysis || [];
  const allSkills = [...required, ...niceToHave];

  if (allSkills.length === 0) return null;

  // Build radar data: each skill gets resume score + github score
  const data = allSkills.slice(0, 12).map((skill) => ({
    skill: skill.skill.length > 12 ? skill.skill.slice(0, 12) + "…" : skill.skill,
    resume: skill.in_resume ? 100 : 0,
    github: skill.in_github ? 100 : 0,
    confidence: skill.evidence_strength != null ? Math.round(skill.evidence_strength) : skill.probability != null ? Math.round(skill.probability * 100) : 0,
  }));

  return (
    <div className="skill-radar-chart">
      <h3>Skill Coverage Radar</h3>
      <ResponsiveContainer width="100%" height={350}>
        <RadarChart data={data} cx="50%" cy="50%" outerRadius="70%">
          <PolarGrid stroke="var(--color-border, #e2e8f0)" />
          <PolarAngleAxis
            dataKey="skill"
            tick={{ fill: "var(--text-light, #64748b)", fontSize: 11 }}
          />
          <PolarRadiusAxis angle={30} domain={[0, 100]} tick={false} />
          <Radar
            name="Resume"
            dataKey="resume"
            stroke="#2563eb"
            fill="#2563eb"
            fillOpacity={0.25}
          />
          <Radar
            name="GitHub"
            dataKey="github"
            stroke="#06b6d4"
            fill="#06b6d4"
            fillOpacity={0.25}
          />
          <Radar
            name="ML Confidence"
            dataKey="confidence"
            stroke="#8b5cf6"
            fill="#8b5cf6"
            fillOpacity={0.15}
          />
          <Legend />
        </RadarChart>
      </ResponsiveContainer>
    </div>
  );
}

export default SkillRadarChart;
