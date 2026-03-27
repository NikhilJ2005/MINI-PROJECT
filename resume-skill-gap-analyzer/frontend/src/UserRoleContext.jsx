import { createContext, useContext, useState, useCallback } from "react";

const UserRoleContext = createContext(null);

export function UserRoleProvider({ children }) {
  const [userRole, setUserRoleState] = useState(
    () => localStorage.getItem("userRole") || null
  );

  const setUserRole = useCallback((role) => {
    setUserRoleState(role);
    localStorage.setItem("userRole", role);
  }, []);

  const switchRole = useCallback(() => {
    setUserRoleState(null);
    localStorage.removeItem("userRole");
  }, []);

  // Keep logout as an alias for backward compatibility
  const logout = switchRole;

  return (
    <UserRoleContext.Provider value={{ userRole, setUserRole, logout, switchRole }}>
      {children}
    </UserRoleContext.Provider>
  );
}

export function useUserRole() {
  const ctx = useContext(UserRoleContext);
  if (!ctx) throw new Error("useUserRole must be used within UserRoleProvider");
  return ctx;
}
