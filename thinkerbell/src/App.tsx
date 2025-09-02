import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { Layout } from './components/Layout';
import Dashboard from './pages/Dashboard';
import TemplatePage from './pages/TemplatePage';

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      retry: 2,
      staleTime: 5 * 60 * 1000,
      refetchOnWindowFocus: false,
    },
  },
});

function SimpleDashboard() {
  return (
    <div className="space-y-6 p-6">
      <h1 className="text-4xl font-black text-black">
        Thinkerbell <span className="text-pink-600">⚡</span>
      </h1>
      <div className="bg-white rounded-lg shadow-lg p-6">
        <h2 className="text-xl font-bold text-green-600 mb-4">✅ System Working</h2>
        <p>Functionality will be restored shortly...</p>
      </div>
    </div>
  );
}

function App() {
  console.log('🎯 Clean App rendering...');
  
  try {
    return (
      <QueryClientProvider client={queryClient}>
        <Router>
          <Layout>
            <Routes>
              <Route path="/" element={<Dashboard />} />
              <Route path="/model" element={<SimpleDashboard />} />
              <Route path="/template" element={<TemplatePage />} />
              <Route path="/examples" element={<SimpleDashboard />} />
              <Route path="*" element={<Dashboard />} />
            </Routes>
          </Layout>
        </Router>
      </QueryClientProvider>
    );
  } catch (error) {
    console.error('❌ App component error:', error);
    return (
      <div className="min-h-screen flex items-center justify-center bg-gray-100">
        <div className="text-center p-8 bg-white rounded-lg shadow-lg">
          <h1 className="text-2xl font-bold text-red-600 mb-4">Application Error</h1>
          <p className="text-gray-600 mb-4">{String(error)}</p>
          <button 
            onClick={() => window.location.reload()}
            className="px-4 py-2 bg-pink-600 text-white rounded hover:bg-pink-700"
          >
            Reload App
          </button>
        </div>
      </div>
    );
  }
}

export default App;