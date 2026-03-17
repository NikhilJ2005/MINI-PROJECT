import { useState } from "react";
import "./cssFile/GithubUpload.css";

const GithubUpload = ({onUserNameEnter}) => {
  const [username, setUsername] = useState("");

  const handleChange=(e)=>{
    const value=e.target.value;
    // GitHub usernames: alphanumeric/hyphens, no leading/trailing hyphen, max 39 chars
    const sanitized = value.replace(/[^a-zA-Z0-9-]/g, "").slice(0, 39);
    setUsername(sanitized);
    onUserNameEnter(sanitized);
  };

  return (
    <div className="form-group">
      <label htmlFor="github-username" className="form-label">
        GitHub Username <span style={{fontWeight: "normal", opacity: 0.7}}>(optional — can be auto-detected from resume)</span>
      </label>

      <div className="input-with-prefix">
        <span className="input-prefix">@</span>
        <input
          type="text"
          id="github-username"
          name="github-username"
          placeholder="your-github-username"
          className="text-input"
          value={username}
          onChange={handleChange}
          autoComplete="off"
        />
      </div>
    </div>
  );
};

export default GithubUpload;