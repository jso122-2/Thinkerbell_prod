import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index-simple.css'
import App from './App'

console.log('🚀 Loading Thinkerbell (DEBUG MODE - no MSW)...')

// Skip MSW entirely for debugging
try {
  const rootElement = document.getElementById('root')
  console.log('📍 Root element found:', !!rootElement)
  
  if (rootElement) {
    console.log('🎨 Creating React root...')
    const root = createRoot(rootElement)
    
    console.log('⚡ Rendering App component...')
    root.render(
      <StrictMode>
        <App />
      </StrictMode>
    )
    console.log('✅ App rendered successfully!')
  } else {
    console.error('❌ Root element not found')
  }
} catch (error) {
  console.error('❌ Failed to start Thinkerbell:', error)
  
  // Emergency fallback
  const body = document.body
  if (body) {
    body.innerHTML = `
      <div style="padding: 40px; text-align: center; font-family: sans-serif;">
        <h1 style="color: #FF1493; font-size: 48px;">Thinkerbell ⚡</h1>
        <div style="background: #fef2f2; border: 2px solid #fca5a5; border-radius: 12px; padding: 20px; margin: 20px auto; max-width: 500px;">
          <h2 style="color: #dc2626;">React Error</h2>
          <p>Failed to load: ${error}</p>
          <pre style="background: #f5f5f5; padding: 10px; text-align: left; font-size: 12px;">${error}</pre>
        </div>
      </div>
    `
  }
}
