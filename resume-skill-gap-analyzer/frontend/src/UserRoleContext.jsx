import { createContext, useContext, useState, useCallback } from "react";

const UserRoleContext = createContext(null);

export function UserRoleProvider({ children }) {
  const [userRole, setUserRoleState] = useState(
    () => sessionStorage.getItem("userRole") || null
  );

  const setUserRole = useCallback((role) => {
    setUserRoleState(role);
    sessionStorage.setItem("userRole", role);
  }, []);

  const logout = useCallback(() => {
    setUserRoleState(null);
    sessionStorage.removeItem("userRole");
  }, []);

  return (
    <UserRoleContext.Provider value={{ userRole, setUserRole, logout }}>
      {children}
    </UserRoleContext.Provider>
  );
}

export function useUserRole() {
  const ctx = useContext(UserRoleContext);
  if (!ctx) throw new Error("useUserRole must be used within UserRoleProvider");
  return ctx;
}
