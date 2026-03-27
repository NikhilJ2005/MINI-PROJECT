import { useUserRole } from "./UserRoleContext";
import "./cssFile/NotAuthorized.css";

function NotAuthorized({ message = "Recruiter access required" }) {
  const { switchRole } = useUserRole();

  return (
    <div className="not-authorized">
      <div className="not-authorized-icon">🔒</div>
      <h2 className="not-authorized-title">Access Restricted</h2>
      <p className="not-authorized-message">{message}</p>
      <button className="not-authorized-btn" onClick={switchRole}>
        Switch Role
      </button>
    </div>
  );
}

export default NotAuthorized;
