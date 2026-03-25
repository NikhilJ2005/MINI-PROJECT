import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { UserRoleProvider } from './UserRoleContext'
import './cssFile/index.css'
import App from './App.jsx'

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <UserRoleProvider>
      <App />
    </UserRoleProvider>
  </StrictMode>,
)
